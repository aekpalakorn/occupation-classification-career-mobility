import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Join chunked CSV files into a single output CSV.")
    parser.add_argument("--from_chunk", type=int, required=True, help="Starting chunk index (inclusive)")
    parser.add_argument("--to_chunk", type=int, required=True, help="Ending chunk index (inclusive)")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file")
    parser.add_argument("--chunk_prefix", default="data/output_", help="Prefix for chunk CSV files")
    parser.add_argument("--chunk_suffix", default=".csv", help="Suffix for chunk CSV files (e.g., .csv)")
    args = parser.parse_args()

    all_dfs = []
    for i in range(args.from_chunk, args.to_chunk + 1):
        chunk_path = f"{args.chunk_prefix}{i}{args.chunk_suffix}"
        if not os.path.exists(chunk_path):
            print(f"[WARNING] Chunk file not found: {chunk_path} — skipping.")
            continue
        print(f"[INFO] Reading {chunk_path}")
        try:
            df_chunk = pd.read_csv(chunk_path)
            all_dfs.append(df_chunk)
        except Exception as e:
            print(f"[ERROR] Failed to read {chunk_path}: {e}")

    if not all_dfs:
        print("[ERROR] No valid chunk CSV files found. Exiting.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Successfully merged {len(all_dfs)} chunk(s) into {args.output_csv}")
    print(f"[INFO] Total rows: {len(combined_df)}")

if __name__ == "__main__":
    main()
