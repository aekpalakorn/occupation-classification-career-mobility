#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import re

def has_mojibake(text):
    """Detect typical mojibake patterns."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[âÃ¢€]", text))

def smart_fix(text, max_depth=10):
    """
    Adaptive per-row fixer for mojibake.
    Tries multiple decoding strategies until the string stops changing or max_depth reached.
    """
    if not isinstance(text, str):
        return text

    fixed = text
    for _ in range(max_depth):
        old = fixed
        try:
            # 1. Try cp1252 -> UTF-8
            temp = fixed.encode('cp1252', errors='replace').decode('utf-8', errors='replace')
            if temp != fixed:
                fixed = temp
                continue
        except Exception:
            pass

        try:
            # 2. Try Latin1 -> UTF-8 (for Ã / accented letters)
            temp = fixed.encode('latin1', errors='replace').decode('utf-8', errors='replace')
            if temp != fixed:
                fixed = temp
                continue
        except Exception:
            pass

        try:
            # 3. Attempt UTF-8 re-interpretation (for CJK or already-UTF-8 multi-byte)
            temp = fixed.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            if temp != fixed:
                fixed = temp
                continue
        except Exception:
            pass

        # If nothing changed, stop
        if fixed == old:
            break

    return fixed

def fix_column(col_series):
    """Apply adaptive fixer to entire column."""
    return col_series.astype(str).apply(smart_fix)

def check_mojibake(df):
    """Return dict: column -> list of (row_index, string) still containing mojibake."""
    result = {}
    for col in df.select_dtypes(include=[object]).columns:
        problematic_rows = df[df[col].apply(has_mojibake)][col]
        if not problematic_rows.empty:
            result[col] = list(problematic_rows.items())
    return result

def fix_csv(input_file, output_file):
    # Load CSV
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"CSV loaded successfully. Shape: {df.shape}")

    # Fix each string column
    for col in df.select_dtypes(include=[object]).columns:
        if df[col].apply(has_mojibake).any():
            print(f"Fixing mojibake in column: {col}")
            df[col] = fix_column(df[col])

    # Check remaining mojibake
    mojibake_report = check_mojibake(df)
    if mojibake_report:
        print("\nColumns still containing potential mojibake and example strings:")
        for col, rows in mojibake_report.items():
            print(f"\nColumn: {col} ({len(rows)} rows still affected)")
            for idx, s in rows[:10]:
                print(f"Row {idx}: {s!r}")
                print(f"  UTF-8 bytes: {s.encode('utf-8', errors='replace')}")
            if len(rows) > 10:
                print(f"  ...and {len(rows)-10} more rows")
    else:
        print("\nNo mojibake detected. All string columns appear clean!")

    # Save CSV as UTF-8
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nCSV saved successfully as UTF-8: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix CSV mojibake with adaptive per-row decoding (cp1252/Latin1/UTF-8)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", required=True, help="Output fixed CSV file path")
    args = parser.parse_args()

    fix_csv(args.input, args.output)
