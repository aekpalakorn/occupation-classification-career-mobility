import argparse
import pandas as pd
import os

def find_missing_rows(source_csv, target_csv, output_csv):
    # Read source and target
    source_df = pd.read_csv(source_csv, dtype={"task_id": str})
    target_df = pd.read_csv(target_csv, dtype={"task_id": str})

    # Ensure required columns exist
    if "task_id" not in source_df.columns:
        raise ValueError(f"'task_id' column not found in {source_csv}")
    if "task_id" not in target_df.columns:
        raise ValueError(f"'task_id' column not found in {target_csv}")

    # Convert to sets for fast diff
    source_ids = set(source_df["task_id"].astype(str))
    target_ids = set(target_df["task_id"].astype(str))

    missing_ids = source_ids - target_ids

    print(f"[INFO] Total source rows: {len(source_ids)}")
    print(f"[INFO] Total target rows: {len(target_ids)}")
    print(f"[INFO] Missing rows: {len(missing_ids)}")

    # Filter missing rows
    missing_rows_df = source_df[source_df["task_id"].astype(str).isin(missing_ids)]

    # Select columns to output
    columns_to_output = ["task_id", "sentence"]
    missing_rows_df = missing_rows_df[[col for col in columns_to_output if col in missing_rows_df.columns]]

    # Save diff file
    missing_rows_df.to_csv(output_csv, index=False)
    print(f"[INFO] Missing rows written to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Find missing rows between source and target CSVs based on task_id.")
    parser.add_argument("--source_csv", required=True, help="Path to the original full source CSV")
    parser.add_argument("--target_csv", required=True, help="Path to the joined prediction CSV")
    parser.add_argument("--output_csv", default="diff.csv", help="Path to output diff CSV containing missing rows")
    args = parser.parse_args()

    if not os.path.exists(args.source_csv):
        raise FileNotFoundError(f"Source file not found: {args.source_csv}")
    if not os.path.exists(args.target_csv):
        raise FileNotFoundError(f"Target file not found: {args.target_csv}")

    find_missing_rows(args.source_csv, args.target_csv, args.output_csv)

if __name__ == "__main__":
    main()
