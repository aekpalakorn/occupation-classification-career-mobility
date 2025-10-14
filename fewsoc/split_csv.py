import argparse
import pandas as pd
import os
import math

def split_csv(input_csv, output_dir, n_chunks):
    # Load CSV
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"Total rows in input CSV: {total_rows}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate chunk size (ceil to ensure all rows are included)
    chunk_size = math.ceil(total_rows / n_chunks)
    print(f"Splitting into {n_chunks} chunks (~{chunk_size} rows per chunk)")

    # Split and save each chunk
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_df = df.iloc[start_idx:end_idx]

        output_file = os.path.join(output_dir, f"chunk_{i+1}.csv")
        chunk_df.to_csv(output_file, index=False)
        print(f"Saved chunk {i+1}: rows {start_idx}-{end_idx-1} -> {output_file}")

    print("CSV split complete.")

def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into n chunks.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save CSV chunks")
    parser.add_argument("--n_chunks", type=int, required=True, help="Number of chunks to create")
    args = parser.parse_args()

    split_csv(args.input_csv, args.output_dir, args.n_chunks)

if __name__ == "__main__":
    main()
