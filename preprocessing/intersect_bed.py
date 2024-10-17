import subprocess
import pandas as pd
import os
import argparse

bedtools_path = "/opt/homebrew/bin/bedtools"

def get_column_header(bed_file):
    with open(bed_file, 'r') as f:
        header = f.readline().strip().split()
    return header 
    
def get_column_number(bed_file):
    with open(bed_file, 'r') as f:
        header = f.readline().strip().split()
    return len(header)
    
def intersect_bed_files(bed_files, output_folder):
    if len(bed_files) < 2:
        raise ValueError("At least two BED files are required for intersection.")
    
    for bed_file in bed_files:
        if not os.path.isfile(bed_file):
            raise FileNotFoundError(f"{bed_file} does not exist.")

    current_intersection = bed_files[0]
    sum_nb_col = 0
    cols_bed = [get_column_number(bed_files[0])]
    total_header = get_column_header(bed_files[0])
    
    for bed_file in bed_files[1:]:
        header = get_column_header(bed_file)
        total_header += header
        cols_bed.append(get_column_number(bed_files[-1]))
        sum_nb_col += cols_bed[-1]
        output_intersection = f"intermediate_intersection.bed"
        
        # Run bedtools intersect
        intersect_cmd = [
            bedtools_path, "intersect", 
            "-a", current_intersection, 
            "-b", bed_file, 
            "-wa", "-wb"
        ]
        
        intersect_result = subprocess.run(intersect_cmd, capture_output=True, text=True)

        with open(output_intersection, 'w') as f:
            f.write(intersect_result.stdout)

        current_intersection = output_intersection

    df = pd.read_csv(current_intersection, sep='\t', header=None)

    if (df[2] - df[1] >= 2).any():
        cols_sum = {}
    
        for k in range(len(total_header)):
            if total_header[k] == "count_modified" or total_header[k] == "valid_coverage":
                cols_sum[k] = "sum"
            else:
                cols_sum[k] = "first"

        df = df.groupby([0, 1, 2], as_index=False).agg(cols_sum)

    start_idx = 0
    for i, bed_file in enumerate(bed_files):
        end_idx = start_idx + cols_bed[i]
        df_selected = df.iloc[:, start_idx:end_idx]
        df_selected.columns = total_header[start_idx:end_idx]
        output_file = f"bed{i + 1}_intersect.bed"

        if i >= 1:
            df_selected['percent_modified'] = (df_selected['count_modified'] / df_selected['valid_coverage']) * 100
            
        df_selected.to_csv(output_folder + "/" + output_file, sep='\t', header=True, index=False)
        start_idx = end_idx

    print(f"Intersected files created: {[f'{bed_file}_intersect.bed' for bed_file in bed_files]}")

def main():
    parser = argparse.ArgumentParser(description="Intersect multiple BED files using bedtools.")
    parser.add_argument('bed_files', nargs='+', help="List of BED files to intersect (at least two files required).")
    parser.add_argument('out', nargs='?', type=str, default=".", help='Path to output folder')
    
    args = parser.parse_args()

    output_folder = os.path.join(os.getcwd(), args.output_folder)
    if not os.path.exists(output_folder):
        print(f'Creating directory {output_folder} to store results')
        os.mkdir(output_folder)
    
    intersect_bed_files(args.bed_files, output_folder)

if __name__ == "__main__":
    main()
