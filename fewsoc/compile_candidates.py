import pandas as pd
import os
import argparse
from typing import List

def prepare_data_for_final_selection(input_files: List[str], output_file: str, sentence_column: str = 'sentence'):
    """
    Reads prediction files from multiple models, compiles all unique predicted 
    titles and codes for each input sentence, and saves the results to a single CSV.

    Args:
        input_files: A list of paths to the input CSV prediction files.
        output_file: The path to the output CSV file.
        sentence_column: The name of the column containing the input sentence/job title.
    """
    print(f"[INFO] Starting data preparation process...")
    
    # List to store DataFrame slices from each file
    all_df_slices = []
    
    # Column names expected in the input prediction files
    required_cols = ['pred_soc_title', 'pred_soc_code', sentence_column]
    
    # --- CONSTANTS FOR CLEANING ---
    INVALID_SOC_CODE = '00-0000.00'
    INVALID_TITLE_STRINGS = {'nan', 'none', 'n/a', 'unknown', 'not applicable'}
    # ------------------------------
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"[WARNING] Input file not found, skipping: {file_path}")
            continue

        try:
            print(f"[INFO] Processing file: {file_path}")
            df_pred = pd.read_csv(file_path)

            # --- CHECK FOR REQUIRED COLUMNS ---
            missing_cols = [col for col in required_cols if col not in df_pred.columns]
            if missing_cols:
                print(f"[WARNING] Skipping {file_path}. Missing required columns: {missing_cols}")
                continue

            # 1. Create the combined label string: "Title (Code)" with strict filtering
            df_pred['combined_label'] = df_pred.apply(
                lambda row: f"{row['pred_soc_title']} ({row['pred_soc_code']})" 
                if (
                    # A. Check for valid SOC Code (string, not empty, not the dummy code)
                    isinstance(row['pred_soc_code'], str) and 
                    row['pred_soc_code'].strip() and 
                    row['pred_soc_code'].strip() != INVALID_SOC_CODE and
                    
                    # B. Check for valid Title (string, not empty, not a placeholder string)
                    isinstance(row['pred_soc_title'], str) and 
                    row['pred_soc_title'].strip() and 
                    row['pred_soc_title'].strip().lower() not in INVALID_TITLE_STRINGS
                )
                else None,
                axis=1
            )
            
            # 2. Drop rows where combined_label creation failed (i.e., invalid code or invalid title)
            df_candidates = df_pred.dropna(subset=['combined_label']).copy()

            # 3. Extract only the sentence and the combined label
            df_candidates = df_candidates[[sentence_column, 'combined_label']]
            
            # Remove duplicate predictions within the current model's output
            df_candidates.drop_duplicates(inplace=True)

            # 4. Append to the list
            all_df_slices.append(df_candidates)
            
            print(f"[INFO] Extracted {len(df_candidates)} unique candidates from {file_path}.")

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    if not all_df_slices:
        print("[FATAL] No valid data was processed from any input file. Exiting.")
        return

    # 5. Concatenate ALL candidate slices into a single master DataFrame
    master_candidates = pd.concat(all_df_slices, ignore_index=True)
    
    print(f"[INFO] Total candidate predictions compiled: {len(master_candidates)}")
    
    # 6. Group by the input sentence and aggregate all unique combined_label strings
    grouped_candidates = master_candidates.groupby(sentence_column)['combined_label'].agg(
        lambda x: sorted(list(set(x)))
    ).reset_index(name='labels_list')

    # 7. Format the final output: join the list of unique labels with a semi-colon
    grouped_candidates['labels'] = grouped_candidates['labels_list'].apply(
        lambda labels: ";".join(labels)
    )
    
    # 8. Select the final required columns: 'sentence' and 'labels'
    final_df = grouped_candidates[[sentence_column, 'labels']]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 9. Save the final compiled data
    final_df.to_csv(output_file, index=False)
    
    print(f"[SUCCESS] Data preparation complete. Compiled {len(final_df)} unique sentences.")
    print(f"[SUCCESS] Final candidates saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile SOC predictions from multiple model outputs for final selection.")
    parser.add_argument("--input_files", 
                        nargs='+', 
                        required=True, 
                        help="List of paths to input prediction CSV files.")
    parser.add_argument("--output_file", 
                        required=True, 
                        help="Path to the final output CSV file.")
    parser.add_argument("--sentence_column",
                        default='sentence',
                        help="Name of the column containing the input text/sentence (default: 'sentence').")
    
    args = parser.parse_args()
    
    prepare_data_for_final_selection(
        args.input_files, 
        args.output_file, 
        args.sentence_column
    )