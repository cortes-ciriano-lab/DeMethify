import pandas as pd
import os
import argparse
import numpy as np
from scipy.linalg import svd

def feature_select(bedfile, n, output_folder, method="var"):
    df = pd.read_csv(bedfile, sep='\t')
    df_cleaned = df.dropna()

    if method == "var":
        row_variances = df_cleaned.iloc[:, 3:].var(axis=1)
        selected_rows = df_cleaned.loc[row_variances.nlargest(n).index]
    elif method == "svd":
        A = df_cleaned.iloc[:, 3:].values
        U, _, _ = svd(A, full_matrices=False)
        row_scores = np.abs(U[:, :n]).sum(axis=1)
        selected_rows = df_cleaned.iloc[np.argsort(-row_scores)[:n]]
    else:
        raise ValueError("Invalid method! Choose 'var' or 'svd'.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, os.path.basename(bedfile).replace(".bed", "_select_ref.bed"))
    selected_rows.to_csv(output_file, sep='\t', header=True, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select top N rows using variance or SVD from a BED file.')
    parser.add_argument('--bed', type=str, required=True, help='Path to the input BED file')
    parser.add_argument('--n', type=int, required=True, help='Number of top rows to select')
    parser.add_argument('--out', nargs='?', type=str, default=".", help='Path to output folder')
    parser.add_argument('--method', type=str, choices=["var", "svd"], default="var")

    args = parser.parse_args()
    feature_select(args.bed, args.n, args.out, args.method)

