import argparse
import pandas as pd
import math
from typing import Dict, Set, List, Tuple

# --- Utility Functions ---

def tokenize_and_clean(text: str) -> List[str]:
    """Tokenizes and cleans a single string."""
    return text.lower().replace(',', '').split()

def build_inverted_index(canonical_titles: Set[str]) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    """
    Builds an inverted index mapping tokens to canonical titles containing them.
    Also returns a pre-tokenized map for canonical titles.
    """
    inverted_index: Dict[str, Set[str]] = {}
    canonical_token_map: Dict[str, List[str]] = {}
    
    for title in canonical_titles:
        tokens = tokenize_and_clean(title)
        canonical_token_map[title] = tokens
        for token in set(tokens):
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(title)

    return inverted_index, canonical_token_map

# --- Postprocess titles with Deterministic Tie-Breaker ---

def postprocess_titles(onet_csv, pred_csv, output_csv):
    
    # Load canonical SOC titles and codes
    onet_df = pd.read_csv(onet_csv)
    if "code" not in onet_df.columns or "title" not in onet_df.columns:
        raise ValueError("The onet_csv must contain 'code' and 'title' columns.")
    
    onet_df["title"] = onet_df["title"].fillna("")
    
    canonical_set = set(t for t in onet_df["title"].tolist() if t) 
    title_to_code = {row["title"]: row["code"] for _, row in onet_df.iterrows() if row["title"]}

    print(f"[INFO] Loaded {len(canonical_set)} canonical SOC titles (non-empty).")

    # Build Inverted Index (for O(M + N + Search) efficiency)
    print("[INFO] Building inverted index...")
    inverted_index, canonical_token_map = build_inverted_index(canonical_set)
    print(f"[INFO] Inverted index built with {len(inverted_index)} unique tokens.")
    
    # Load prediction CSV
    pred_df = pd.read_csv(pred_csv)
    if "pred_soc_title" not in pred_df.columns or "pred_soc_code" not in pred_df.columns:
        raise ValueError("The prediction CSV must contain 'pred_soc_title' and 'pred_soc_code' columns.")
    
    pred_df["pred_soc_title"] = pred_df["pred_soc_title"].fillna("")
    print(f"[INFO] Loaded {len(pred_df)} predictions.")

    corrected_titles = []
    corrected_codes = []

    num_fixed = 0
    num_unmapped = 0
    
    pred_titles = pred_df["pred_soc_title"].tolist()
    
    for pred_title in pred_titles:
        
        if not pred_title:
             corrected_titles.append("None")
             corrected_codes.append("00-0000.00")
             num_unmapped += 1
             continue
        
        # 1. Direct match check
        if pred_title in canonical_set:
            corrected_titles.append(pred_title)
            corrected_codes.append(title_to_code[pred_title])
        else:
            # 2. Similarity search using Inverted Index
            pred_tokens = tokenize_and_clean(pred_title)
            
            if not pred_tokens:
                 corrected_titles.append("None")
                 corrected_codes.append("00-0000.00")
                 num_unmapped += 1
                 continue
                 
            candidate_titles: Set[str] = set()
            for token in set(pred_tokens):
                if token in inverted_index:
                    candidate_titles.update(inverted_index[token])

            max_sim = 0.0
            # Use a list to store all titles that achieve the current max_sim
            tied_titles: List[str] = []
            
            for canonical_title in candidate_titles: 
                canonical_tokens = canonical_token_map[canonical_title]
                intersection_size = len(set(pred_tokens) & set(canonical_tokens))
                sim = intersection_size / len(pred_tokens)
                
                if sim > max_sim:
                    # New best score found
                    max_sim = sim
                    tied_titles = [canonical_title] # Reset and start new list
                elif sim == max_sim and sim > 0:
                    # Tie found at the current max score (sim > 0 check is redundant due to inverted index, but safer)
                    tied_titles.append(canonical_title)

            # --- Apply Deterministic Tie-Breaker (Alphabetical Sort) ---
            if tied_titles:
                # Sort the tied candidates alphabetically to ensure a deterministic choice
                tied_titles.sort()
                
                best_title = tied_titles[0]
                
                corrected_titles.append(best_title)
                corrected_codes.append(title_to_code[best_title])
                num_fixed += 1
            else:
                # 3. No match found (max_sim was 0 or no candidates)
                corrected_titles.append("None")
                corrected_codes.append("00-0000.00")
                num_unmapped += 1

    # Overwrite the original columns
    pred_df["pred_soc_title"] = corrected_titles
    pred_df["pred_soc_code"] = corrected_codes

    print(f"[INFO] Fixed {num_fixed} hallucinated titles.")
    print(f"[INFO] {num_unmapped} titles could not be mapped to any canonical title or were null.")
    print(f"[INFO] Postprocessed file written to {output_csv}")

    pred_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process predicted SOC titles to match canonical O*NET titles.")
    parser.add_argument("--onet_csv", required=True, help="Path to onet-soc_2019.csv")
    parser.add_argument("--pred_csv", required=True, help="Path to predictions CSV")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")

    args = parser.parse_args()
    postprocess_titles(args.onet_csv, args.pred_csv, args.output_csv)