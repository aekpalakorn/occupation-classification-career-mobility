import argparse
import pandas as pd
import json
import time
import os
import sys

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
import requests

# --- GOOGLE AUTH IMPORTS (For Vertex AI Llama) ---
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
# ---------------------------

# --- NEW GEMINI IMPORTS ---
from google import genai
from google.genai import types as genai_types
# --------------------------

# Define the model identifier constants
VERTEX_LLAMA_MODEL_PREFIX = "meta/llama-3.1"
GEMINI_MODEL_PREFIX = "gemini-2.5"

# Initialize global client placeholders
openai_client = None
gemini_client = None

# --- HELPER FUNCTION TO GET TOKEN FROM SA FILE (Unchanged for Vertex AI Llama) ---
def get_gcp_access_token():
    """Fetches a fresh GCP access token using environment credentials (SA or ADC)."""
    
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    
    try:
        credentials, project_id = google_auth_default(scopes=SCOPES)
        
        if not project_id:
              raise ValueError("Project ID could not be determined from credentials. Check file contents or IAM role.")
            
        auth_request = GoogleAuthRequest()
        credentials.refresh(auth_request)
        
        return credentials.token, project_id
        
    except Exception as e:
        import traceback
        error_info = "".join(traceback.format_exc())
        print(f"[FATAL_AUTH_ERROR] Authentication failed. Details:\n{error_info}", file=sys.stderr)
        
        raise RuntimeError(f"GCP Authentication failed: {e}")

# --- REVISED call_api FUNCTION with Gemini Support ---
@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_api(model: str, prompt: str, temperature: float = 0, project_id: str = None, region: str = "us-central1"):
    """Call OpenAI, Vertex AI Chat API, or Gemini API with retries."""
    
    system_instruction = "You are an expert ONET-SOC 2019 coder. Classify jobs into SOC codes."
    
    if model.startswith("gpt"):
        # --- OpenAI API Call ---
        global openai_client
        if openai_client is None:
            openai_client = OpenAI()
            
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            usage = getattr(response.usage, "to_dict", lambda: response.usage)
            return {
                "choices": [{"message": {"content": response.choices[0].message.content}}],
                "usage": usage()
            }
        except Exception as e:
            raise Exception(f"OpenAI API Call Error: {e}")
            
    elif GEMINI_MODEL_PREFIX in model.lower():
        # --- Google Gemini API Call (Using google-genai SDK) ---
        global gemini_client
        # Client initialized in main, ensure it exists
        if gemini_client is None: 
            raise RuntimeError("Gemini client not initialized. Check API key setup.")
            
        try:
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )
            
            response = gemini_client.models.generate_content(
                model=model,
                contents=[prompt],
                config=config
            )
            
            # Standardize the return structure
            usage_metadata = response.usage_metadata
            return {
                "choices": [{"message": {"content": response.text}}],
                "usage": {
                    "prompt_tokens": usage_metadata.prompt_token_count,
                    "completion_tokens": usage_metadata.candidates_token_count,
                    "total_tokens": usage_metadata.total_token_count
                }
            }
        except Exception as e:
            raise Exception(f"Gemini API Call Error: {e}")
            
    elif VERTEX_LLAMA_MODEL_PREFIX in model.lower():
        # --- Vertex AI Llama 3.1 REST API Call ---
        
        token, sa_project_id = get_gcp_access_token()
        
        final_project_id = project_id if project_id else sa_project_id
        
        if not final_project_id or not region:
            raise ValueError("Vertex AI Llama model requires a project_id and region.")

        MAAS_ENDPOINT_PATH = "openapi/chat/completions"
        endpoint_url = (
            f"https://{region}-aiplatform.googleapis.com/v1/projects/{final_project_id}/"
            f"locations/{region}/endpoints/{MAAS_ENDPOINT_PATH}"
        )
        
        full_prompt = f"{system_instruction}\n\n{prompt}"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": temperature,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_json = response.json()
            
            model_output = response_json["choices"][0]["message"]["content"]
            usage_data = response_json.get("usage", {})
            
            return {
                "choices": [{"message": {"content": model_output}}],
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens"),
                    "completion_tokens": usage_data.get("completion_tokens"),
                    "total_tokens": usage_data.get("total_tokens")
                }
            }
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                error_details = f"HTTP Status {e.response.status_code}: {e.response.text}"
            else:
                error_details = str(e)
            
            print(f"[FATAL_VERTEX_ERROR] Request Failed: {error_details}")
            raise Exception(f"Vertex AI HTTP Call Error: {error_details}")
            
    else:
        raise ValueError(f"Unsupported model: {model}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--raw_output_json", required=True)
    parser.add_argument("--log_file", required=True)
    
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name (e.g., 'gpt-4o-mini', 'meta/llama-3.1-8b-instruct-maas', or 'gemini-2.5-flash')")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    
    parser.add_argument("--vertex_project", type=str, default=os.environ.get("GCP_PROJECT_ID"), help="GCP Project ID for Vertex AI (optional, read from SA if not set)")
    parser.add_argument("--vertex_location", type=str, default=os.environ.get("GCP_REGION", "us-central1"), help="GCP Region for Vertex AI")
    
    args = parser.parse_args()

    # --- Conditional Client Initialization ---
    client_to_close = None
    
    if args.model.startswith("gpt"):
        # OpenAI Setup
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING] OPENAI_API_KEY not set")
            
        global openai_client
        openai_client = OpenAI()
        
    elif GEMINI_MODEL_PREFIX in args.model.lower():
        # Gemini Setup
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print(f"[DEBUG] Using Gemini API key: {api_key[:4]}...{api_key[-4:]}")
            global gemini_client
            try:
                gemini_client = genai.Client()
                client_to_close = gemini_client # Mark client for explicit closure
                print(f"[INFO] Configuring for Gemini API client for model: {args.model}")
            except Exception as e:
                print(f"[FATAL] Failed to initialize Gemini Client. Error: {e}")
                return
        else:
             print("[FATAL] GEMINI_API_KEY not set. Cannot use Gemini models.")
             return
            
    elif VERTEX_LLAMA_MODEL_PREFIX in args.model.lower():
        # Vertex AI Setup (using requests/google-auth)
        print(f"[INFO] Configuring for Vertex AI Llama REST client in region: {args.vertex_location}")
        
    else:
          print(f"[ERROR] Unsupported model type: {args.model}")
          return

    # Load prompt template and input CSV
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    df = pd.read_csv(args.input_csv)
    df = df.iloc[args.start_row:].reset_index(drop=True)
    total_rows = len(df)
    print(f"[INFO] Starting from row {args.start_row}, total rows to process: {total_rows}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # If output CSV exists, append to it
    output_exists = os.path.exists(args.output_csv)

    # Use a try/finally block to ensure the client is closed
    try:
        # Open raw output JSONL and log file
        with open(args.raw_output_json, "a", encoding="utf-8") as raw_file, \
             open(args.log_file, "a", encoding="utf-8") as log_file:

            # Pre-warm the model
            print("[INFO] Pre-warming the model...")
            try:
                _ = call_api(
                    args.model, 
                    "Pre-warm request. No output needed.", 
                    temperature=args.temperature, 
                    project_id=args.vertex_project, 
                    region=args.vertex_location
                )
                print("[INFO] Model pre-warmed successfully.")
            except Exception as e:
                print(f"[WARNING] Pre-warm request failed: {e}")
                
            # Process batches
            for i in range(0, total_rows, args.batch_size):
                batch = df.iloc[i:i + args.batch_size]
                input_texts = "\n".join(f"{row['task_id']}; {row['sentence']}" for _, row in batch.iterrows())
                prompt = prompt_template.replace("{{data}}", input_texts)

                start_time = time.time()
                response = None
                try:
                    response = call_api(
                        args.model, 
                        prompt, 
                        temperature=args.temperature, 
                        project_id=args.vertex_project, 
                        region=args.vertex_location
                    )
                except Exception as e:
                    error_msg = {
                        "level": "ERROR",
                        "batch_start_row": i + args.start_row,
                        "batch_size": len(batch),
                        "message": str(e)
                    }
                    print(json.dumps(error_msg))
                    log_file.write(json.dumps(error_msg) + "\n")
                    log_file.flush()
                    continue
                batch_latency = time.time() - start_time

                # Token extraction
                prompt_tokens = response['usage'].get("prompt_tokens")
                completion_tokens = response['usage'].get("completion_tokens")
                total_tokens = response['usage'].get("total_tokens")

                # Save raw response
                raw_file.write(json.dumps({
                    "task_id_batch": list(batch["task_id"]),
                    "api_response": response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_sec": batch_latency
                }) + "\n")
                raw_file.flush()

                # Parse model output
                batch_results = []
                model_output = response['choices'][0]['message']['content'].strip()
                
                for line in model_output.splitlines():
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) != 4:
                        warning_msg = {
                            "level": "WARNING",
                            "line": line,
                            "message": "Skipping malformed line"
                        }
                        print(json.dumps(warning_msg))
                        log_file.write(json.dumps(warning_msg) + "\n")
                        log_file.flush()
                        continue

                    task_id, label1, label2, label3 = parts
                    # Attempt to split SOC title and code
                    if ":" in label1:
                        soc_title, soc_code = [p.strip() for p in label1.split(":", 1)]
                    else:
                        soc_title, soc_code = label1.strip(), "" 

                    input_text = batch.loc[batch["task_id"] == task_id, "sentence"].values
                    input_text = input_text[0] if len(input_text) > 0 else ""

                    batch_results.append({
                        "task_id": task_id,
                        "sentence": input_text,
                        "pred_soc_title": soc_title,
                        "pred_soc_code": soc_code,
                        "is_not_occ": label2,
                        "is_multiple": label3
                    })

                # Incrementally write parsed results after each batch
                batch_df = pd.DataFrame(batch_results)
                if not batch_df.empty:
                    # Write header only if the file is new and not empty
                    batch_df.to_csv(args.output_csv, mode="a", index=False, header=not output_exists)
                    output_exists = True

                # Update progress
                info_msg = {
                    "level": "INFO",
                    "processed_rows": i + len(batch) + args.start_row,
                    "total_rows": total_rows,
                    "percent_done": round(((i + len(batch)) / total_rows) * 100, 2),
                    "batch_start_row": i + args.start_row,
                    "batch_size": len(batch),
                    "latency_sec": round(batch_latency, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                print(json.dumps(info_msg))
                log_file.write(json.dumps(info_msg) + "\n")
                log_file.flush()

                time.sleep(0.5)

            # Final success messages
            print(f"[INFO] Finished processing. Parsed results saved to {args.output_csv}")
            print(f"[INFO] Raw API responses saved/appended to {args.raw_output_json}")
            print(f"[INFO] Progress logged to {args.log_file}")
            
    finally:
        # --- FIX FOR SHUTDOWN EXCEPTION ---
        if client_to_close:
            print("[INFO] Explicitly closing Gemini client connections to prevent shutdown error.")
            try:
                client_to_close.close()
            except Exception as e:
                # Log any exception during close, but don't stop shutdown
                print(f"[WARNING] Exception during client close: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()