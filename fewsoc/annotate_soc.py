import argparse
import pandas as pd
import json
import time
import os
import sys

# Reusing imports from classify_soc.py
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
GEMINI_MODEL_PREFIX = "gemini-2.5" # e.g., gemini-2.5-flash

# Initialize global client placeholders
openai_client = None
gemini_client = None

# --- HELPER FUNCTION TO GET TOKEN FROM SA FILE (Copied from classify_soc.py) ---
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

# --- API CALL FUNCTION (Copied from classify_soc.py) ---
@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_api(model: str, prompt: str, temperature: float = 0, project_id: str = None, region: str = "us-central1"):
    """Call OpenAI, Vertex AI Chat API, or Gemini API with retries."""
    
    # NOTE: The system instruction is tailored for the selection task
    system_instruction = "You are an expert ONET-SOC 2019 coder tasked with selecting the most applicable SOC label(s) from a provided list of options for a given job record."
    
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


def parse_final_selection_output(model_output: str) -> str:
    """
    Parses the model's final selection output to extract the semicolon-separated labels.
    Expected format: "Answer: Label1; Label2; ..."
    """
    output = model_output.strip()
    
    # Look for the required prefix "Answer: "
    if output.lower().startswith("answer:"):
        # Strip the prefix and any leading/trailing whitespace
        labels = output[len("Answer:"):].strip()
        return labels
    else:
        # If the model didn't use the prefix, assume the entire output is the label list
        # This is a fallback but suggests prompt following failed
        return output

# --- MAIN EXECUTION LOGIC ---
def main():
    parser = argparse.ArgumentParser(description="Annotate job records by selecting best SOC labels from a list of candidates.")
    parser.add_argument("--prompt_file", required=True, help="Path to the prompt template file.")
    parser.add_argument("--input_csv", required=True, help="Input CSV with 'sentence' and 'labels' columns (semicolon-separated candidates).")
    parser.add_argument("--output_csv", required=True, help="Output CSV to store the final selected labels.")
    parser.add_argument("--raw_output_json", required=True, help="JSONL file to store raw API responses.")
    parser.add_argument("--log_file", required=True, help="File to log processing progress.")
    
    parser.add_argument("--model", default="gpt-4o", help="Model name (e.g., 'gpt-4o', 'gemini-2.5-pro', etc.).")
    parser.add_argument("--start_row", type=int, default=0, help="Starting row index to process.")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature for the model.")
    
    parser.add_argument("--vertex_project", type=str, default=os.environ.get("GCP_PROJECT_ID"), help="GCP Project ID for Vertex AI (optional).")
    parser.add_argument("--vertex_location", type=str, default=os.environ.get("GCP_REGION", "us-central1"), help="GCP Region for Vertex AI.")
    
    args = parser.parse_args()
    
    # --- Client Initialization (Copied from classify_soc.py) ---
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
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"[FATAL] Prompt file not found: {args.prompt_file}")
        return

    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"[FATAL] Input CSV file not found: {args.input_csv}")
        return
        
    df = df.iloc[args.start_row:].reset_index(drop=True)
    total_rows = len(df)
    print(f"[INFO] Starting from row {args.start_row}, total rows to process: {total_rows}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # If output CSV exists, append to it
    output_exists = os.path.exists(args.output_csv)
    
    # Define columns for the final output CSV
    output_columns = ["sentence", "original_candidates", "selected_labels"]

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
                    "Pre-warm request. Select the option: None (00-0000.00).", 
                    temperature=args.temperature, 
                    project_id=args.vertex_project, 
                    region=args.vertex_location
                )
                print("[INFO] Model pre-warmed successfully.")
            except Exception as e:
                print(f"[WARNING] Pre-warm request failed: {e}")
                
            # Process one row (instance) at a time
            batch_size = 1 # Hardcoded for this annotation task
            
            for i in range(0, total_rows, batch_size):
                row_index_in_df = i
                current_row = df.iloc[row_index_in_df]
                
                sentence = current_row["sentence"]
                candidate_labels_semicolon = current_row["labels"]
                
                # Format options for the prompt: one label per line
                if pd.isna(candidate_labels_semicolon) or not candidate_labels_semicolon.strip():
                     # Skip if no valid candidates were provided
                     final_labels = "Error: No candidates provided"
                     response_json = {}
                else:
                    candidate_options_list = candidate_labels_semicolon.split(";")
                    candidate_options_formatted = "\n".join(f"- {label.strip()}" for label in candidate_options_list if label.strip())
                
                    # Build the prompt
                    prompt = prompt_template.replace("{{input}}", sentence)
                    prompt = prompt.replace("{{options}}", candidate_options_formatted)

                    start_time = time.time()
                    response = None
                    final_labels = "API_ERROR"

                    try:
                        response = call_api(
                            args.model, 
                            prompt, 
                            temperature=args.temperature, 
                            project_id=args.vertex_project, 
                            region=args.vertex_location
                        )
                        batch_latency = time.time() - start_time

                        # Parse the final output
                        model_output = response['choices'][0]['message']['content'].strip()
                        final_labels = parse_final_selection_output(model_output)
                        
                        # Token extraction
                        prompt_tokens = response['usage'].get("prompt_tokens")
                        completion_tokens = response['usage'].get("completion_tokens")
                        total_tokens = response['usage'].get("total_tokens")
                        response_json = {
                            "api_response": response,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "latency_sec": batch_latency
                        }

                    except Exception as e:
                        error_msg = {
                            "level": "ERROR",
                            "processed_row": i + args.start_row,
                            "message": str(e)
                        }
                        print(json.dumps(error_msg))
                        log_file.write(json.dumps(error_msg) + "\n")
                        log_file.flush()
                        response_json = {"error": str(e)}
                
                # Save raw response
                raw_file.write(json.dumps({
                    "row_id": i + args.start_row,
                    "sentence": sentence,
                    "candidates": candidate_labels_semicolon,
                    "final_labels": final_labels,
                    **response_json
                }) + "\n")
                raw_file.flush()
                
                # Prepare and write result row
                result_row = pd.DataFrame([{
                    "sentence": sentence,
                    "original_candidates": candidate_labels_semicolon,
                    "selected_labels": final_labels
                }], columns=output_columns)
                
                # Incrementally write parsed results
                result_row.to_csv(args.output_csv, mode="a", index=False, header=not output_exists)
                output_exists = True

                # Update progress
                info_msg = {
                    "level": "INFO",
                    "processed_rows": i + 1 + args.start_row,
                    "total_rows": total_rows,
                    "percent_done": round(((i + 1) / total_rows) * 100, 2),
                    "latency_sec": round(batch_latency, 2) if response else 0,
                }
                print(json.dumps(info_msg))
                log_file.write(json.dumps(info_msg) + "\n")
                log_file.flush()

                time.sleep(0.5)

            # Final success messages
            print(f"[INFO] Finished processing. Selected labels saved to {args.output_csv}")
            print(f"[INFO] Raw API responses saved/appended to {args.raw_output_json}")
            print(f"[INFO] Progress logged to {args.log_file}")
            
    finally:
        # --- FIX FOR SHUTDOWN EXCEPTION ---
        if client_to_close:
            print("[INFO] Explicitly closing Gemini client connections to prevent shutdown error.")
            try:
                client_to_close.close()
            except Exception as e:
                print(f"[WARNING] Exception during client close: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()