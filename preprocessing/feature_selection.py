import pandas as pd
import os
import argparse

def feature_select_var(bedfile, n, output_folder):
    df = pd.read_csv(bedfile, sep='\t')
    
    row_variances = df.iloc[:, 3:].var(axis=1)
    top_n_rows = df.loc[row_variances.nlargest(n).index]

    if not os.path.exists(output_folder):
        print(f'Creating directory {output_folder} to store results')
        os.mkdir(output_folder)
      
    top_n_rows.to_csv(output_folder + "/select_ref.bed", sep='\t', header=True, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select top N rows with highest variance from a BED file.')
    
    parser.add_argument('bedfile', type=str, help='Path to the input BED file')
    parser.add_argument('n', type=int, help='Number of top rows to select')
    parser.add_argument('output_folder', type=str, default=".", help='Path to output folder')

    args = parser.parse_args()

    
    
    feature_select_var(args.bedfile, args.n, args.output_folder)
