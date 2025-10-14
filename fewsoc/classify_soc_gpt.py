import argparse
import pandas as pd
import json
import time
import os
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# Initialize OpenAI client
client = OpenAI()

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_gpt_api(model, prompt, temperature=0):
    """Call OpenAI Chat API with retries."""
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert ONET-SOC 2019 coder. Classify jobs into SOC codes."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--raw_output_json", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    # Print API key (truncated)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"[DEBUG] Using OpenAI API key: {api_key[:4]}...{api_key[-4:]}")
    else:
        print("[WARNING] OPENAI_API_KEY not set")

    # Load prompt template
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Load input CSV
    df = pd.read_csv(args.input_csv)
    df = df.iloc[args.start_row:].reset_index(drop=True)
    total_rows = len(df)
    print(f"[INFO] Starting from row {args.start_row}, total rows to process: {total_rows}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # If output CSV exists, append to it
    output_exists = os.path.exists(args.output_csv)

    # Open raw output JSONL and log file
    with open(args.raw_output_json, "a", encoding="utf-8") as raw_file, \
         open(args.log_file, "a", encoding="utf-8") as log_file:

        processed_rows = 0
        if os.path.exists(args.log_file):
            with open(args.log_file, "r", encoding="utf-8") as f:
                processed_rows = sum(1 for l in f if '"level": "INFO"' in l)

        # Pre-warm the model
        print("[INFO] Pre-warming the model...")
        try:
            _ = call_gpt_api(args.model, "Pre-warm request. No output needed.", temperature=args.temperature)
            print("[INFO] Model pre-warmed successfully.")
        except Exception as e:
            print(f"[WARNING] Pre-warm request failed: {e}")

        # Process batches
        for i in range(0, total_rows, args.batch_size):
            batch = df.iloc[i:i + args.batch_size]
            input_texts = "\n".join(f"{row['task_id']}; {row['sentence']}" for _, row in batch.iterrows())
            prompt = prompt_template.replace("{{data}}", input_texts)

            start_time = time.time()
            try:
                response = call_gpt_api(args.model, prompt, temperature=args.temperature)
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

            # Token usage
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)

            # Save raw response
            raw_file.write(json.dumps({
                "task_id_batch": list(batch["task_id"]),
                "api_response": response.to_dict(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_sec": batch_latency
            }) + "\n")
            raw_file.flush()

            # Parse model output
            batch_results = []
            model_output = response.choices[0].message.content.strip()
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
                batch_df.to_csv(args.output_csv, mode="a", index=False, header=not output_exists)
                output_exists = True

            # Update progress
            processed_rows += len(batch)
            percent_done = (processed_rows / total_rows) * 100
            info_msg = {
                "level": "INFO",
                "processed_rows": processed_rows,
                "total_rows": total_rows,
                "percent_done": round(percent_done, 2),
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

    print(f"[INFO] Finished processing. Parsed results saved to {args.output_csv}")
    print(f"[INFO] Raw API responses saved/appended to {args.raw_output_json}")
    print(f"[INFO] Progress logged to {args.log_file}")

if __name__ == "__main__":
    main()
