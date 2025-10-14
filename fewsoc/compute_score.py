import pandas as pd
import argparse
import os

# --- Preprocessing Function ---

def clean_label(label: str) -> str:
    """
    Standardizes and cleans an SOC label string for comparison.

    Steps:
    1. Check if the input is a valid string.
    2. Convert to lowercase.
    3. Strip leading/trailing whitespace.
    4. Remove commas (,) and parentheses (()).
    5. Strip any resulting extra whitespace again.
    """
    if not isinstance(label, str):
        return ""
    
    cleaned = label.lower().strip()
    cleaned = cleaned.replace(',', '').replace('(', '').replace(')', '')
    return cleaned.strip()

# --- Main Function ---

def compute_accuracy(data_path: str, answer_path: str, answer_label_column: str, output_path: str):
    """
    Computes the accuracy of predicted SOC labels against ground truth answers using 
    the "Any Match" criteria (predicted label must be present in the set of ground-truth labels).
    """
    print(f"[INFO] Loading predicted data from: {data_path}")
    try:
        df_pred = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[FATAL] Predicted data file not found: {data_path}")
        return
        
    print(f"[INFO] Loading ground truth data from: {answer_path}")
    try:
        df_answer = pd.read_csv(answer_path)
    except FileNotFoundError:
        print(f"[FATAL] Ground truth data file not found: {answer_path}")
        return
    
    if answer_label_column not in df_answer.columns:
        print(f"[FATAL] Ground truth column '{answer_label_column}' not found in {answer_path}. Check file columns.")
        return

    # --- 1. Prepare Predicted Label (df_pred) ---
    
    # Create the combined label format: Title (Code)
    df_pred['predicted_label_combined'] = df_pred.apply(
        lambda row: f"{row['pred_soc_title']} ({row['pred_soc_code']})" 
        if pd.notna(row['pred_soc_title']) and pd.notna(row['pred_soc_code']) 
        else '',
        axis=1
    )
    
    # Apply cleaning to the combined predicted label (This will be a single clean string)
    df_pred['predicted_label_cleaned'] = df_pred['predicted_label_combined'].apply(clean_label)
    
    # --- 2. Prepare Ground Truth Labels (df_answer) ---
    
    df_answer = df_answer.rename(columns={answer_label_column: 'ground_truth_label_raw'})
    
    # Convert the semicolon-separated ground truth string into a LIST of CLEANED labels
    def clean_and_split_ground_truth(label_string):
        if not isinstance(label_string, str) or not label_string.strip():
            return []
        # Split by semicolon, clean each individual label, and return as a list
        return [clean_label(lbl) for lbl in label_string.split(';') if clean_label(lbl)]

    df_answer['ground_truth_label_clean_list'] = df_answer['ground_truth_label_raw'].apply(clean_and_split_ground_truth)
    
    # --- 3. Merge DataFrames ---
    
    df_pred_subset = df_pred[['sentence', 'predicted_label_combined', 'predicted_label_cleaned']]
    df_answer_subset = df_answer[['sentence', 'ground_truth_label_raw', 'ground_truth_label_clean_list']]
    
    df_merged = pd.merge(
        df_pred_subset, 
        df_answer_subset, 
        on='sentence', 
        how='inner'
    )
    
    print(f"[INFO] Successfully merged {len(df_merged)} samples based on 'sentence'.")

    # --- 4. Compute Any Match Accuracy ---
    
    # Match condition: Check if the single 'predicted_label_cleaned' is present 
    # in the list of 'ground_truth_label_clean_list'
    def check_any_match(row):
        predicted_label = row['predicted_label_cleaned']
        ground_truth_list = row['ground_truth_label_clean_list']
        
        # Check if the single predicted label (string) is in the list of ground truth labels (list)
        return predicted_label in ground_truth_list

    df_merged['match'] = df_merged.apply(check_any_match, axis=1)
    
    # Calculate accuracy
    total_matched = df_merged['match'].sum()
    total_samples = len(df_merged)
    
    if total_samples == 0:
        accuracy = 0.0
        print("[WARNING] Zero matching samples found after merging. Accuracy is 0.0.")
    else:
        accuracy = total_matched / total_samples
        
    # --- 5. Save Results and Print ---

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_merged.to_csv(output_path, index=False)
    print(f"[INFO] Detailed comparison saved to: {output_path}")

    print("\n" + "="*50)
    print("           ANY MATCH ACCURACY RESULTS")
    print("="*50)
    print(f"Total Samples Compared: {total_samples}")
    print(f"Correctly Matched Samples (Any Match): {total_matched}")
    print(f"Accuracy Score: {accuracy:.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute accuracy score between model predictions and ground truth annotations.")
    parser.add_argument("--data_csv", required=True, help="Path to the model prediction CSV (containing pred_soc_code/title).")
    parser.add_argument("--answer_csv", required=True, help="Path to the ground truth annotation CSV.")
    parser.add_argument("--answer_label_column", default="selected_labels", help="Column name in the answer_csv containing the ground-truth SOC labels (default: 'selected_labels').")
    parser.add_argument("--output_csv", default="accuracy_comparison_results.csv", help="Path to save the merged comparison results.")
    
    args = parser.parse_args()
    
    compute_accuracy(args.data_csv, args.answer_csv, args.answer_label_column, args.output_csv)