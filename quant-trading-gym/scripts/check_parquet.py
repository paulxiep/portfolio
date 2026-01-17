#!/usr/bin/env python3
"""Check parquet file data quality."""
import pyarrow.parquet as pq
import pandas as pd
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/test_recording.parquet'
    t = pq.read_table(path)
    df = t.to_pandas()
    
    print('=== DATA QUALITY CHECK ===')
    print(f'Total rows: {len(df):,}')
    print(f'Total columns: {len(df.columns)}')
    print()
    
    print('=== NULL COUNTS (feature columns) ===')
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    null_counts = df[feature_cols].isnull().sum()
    non_zero_nulls = null_counts[null_counts > 0]
    if len(non_zero_nulls) > 0:
        print(non_zero_nulls)
    else:
        print('No nulls in feature columns!')
    print()
    
    print('=== SAMPLE VALUES (first row) ===')
    first_row = df.iloc[0]
    print(f"tick: {first_row['tick']}")
    print(f"agent_id: {first_row['agent_id']}")
    print(f"f_mid_price: {first_row['f_mid_price']}")
    print(f"f_sma_8: {first_row['f_sma_8']}")
    print(f"f_sma_16: {first_row['f_sma_16']}")
    print(f"f_rsi_8: {first_row['f_rsi_8']}")
    print(f"f_position: {first_row['f_position']}")
    print(f"f_cash: {first_row['f_cash']}")
    print()
    
    print('=== VALUE RANGES ===')
    print(f"f_mid_price: [{df['f_mid_price'].min():.2f}, {df['f_mid_price'].max():.2f}]")
    print(f"f_sma_8: [{df['f_sma_8'].min():.4f}, {df['f_sma_8'].max():.4f}]")
    print(f"f_rsi_8: [{df['f_rsi_8'].min():.2f}, {df['f_rsi_8'].max():.2f}]")
    print(f"f_position: [{df['f_position'].min()}, {df['f_position'].max()}]")

if __name__ == '__main__':
    main()
