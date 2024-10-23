<img width="402" alt="DeMethify" src="https://user-images.githubusercontent.com/79879340/220681790-e3a7edd0-d54c-4a49-b45a-95dca68c44b7.png">

                                       
DeMethify is a partial-reference based methylation deconvolution algorithm that uses a weighted constrained version of an iteratively optimized negative matrix factorization algorithm. 

## Flags and arguments
| Option              | Description                                                                                           |
|---------------------|-------------------------------------------------------------------------------------------------------|
| `methfreq`          | Methylation frequency file path (values between 0 and 1)                                               |
| `ref`               | Methylation reference matrix file path                                                                |
| `outdir`            | Output directory (can exist but must be empty)                                                        |
| `nbunknown`         | Number of unknown cell types to estimate                                                              |
| `iterations`        | Numbers of iterations for outer and inner loops (default = 10000, 20)                                 |
| `termination`       | Termination condition for cost function (default = 1e-2)                                              |
| `init`              | Initialisation option (default = random uniform)                                                      |
| `fillna`            | Replace every NA by 0 in the given data                                                               |
| `ic`                | Select number of unknown cell types by minimising an information criterion (AIC or BIC)               |
| `confidence`        | Outputs bootstrap confidence intervals, takes confidence level and bootstrap iteration numbers as input |
| `plot`              | Plot cell type proportions estimates for each sample, eventually with confidence intervals.            |
| `bedmethyl`         | Flag to indicate that the input will be bedmethyl files, modkit style                                  |
| `counts`            | Read counts file path                                                                                 |
| `noreadformat`      | Flag to use when the data isn't using the read format (e.g., Illumina epic arrays)                    |

## Installing DeMethify

We recommend setting up a fresh conda environment with a Python version >= 3.6 :
```
conda create --name demethify python=3.10.15
conda activate demethify
```

Then one can either use:
```
pip install git+https://github.com/cortes-ciriano-lab/DeMethify
```

Or:
```
git clone https://github.com/cortes-ciriano-lab/DeMethify
cd DeMethify
pip install .
```

Verify that the installation went well with:

```
demethify -h
```


## Run DeMethify

After installing, you can finally run DeMethify. 

The typical pipeline for bedmethyl files (like the ones outputted by modkit) is:
- Preprocessing
  - Potentially feature selection, doeable from commandline with preprocessing/feature_selection.py (see preprocessing/preprocessing.ipynb)
  - Intersection of the reference and the samples so that the CpG sites are consistent across files, doeable from commandline with preprocessing/intersect_bed.py (see preprocessing/preprocessing.ipynb)
- Run DeMethify depending on your use case
```
python feature_selection.py bed1.bed 100000
python intersect_bed.py bed1_select_ref.bed bed2.bed bed3.bed bed4.bed 
```

If you've got a number of samples greater or equal than 2, you can use the partial-reference based algorithm to jointly estimate the unknown cell type portion methylation profile and the proportions of all known and unknown cell types, otherwise you can use the reference based algorithm (if you don't specify --nbunknown) and hope that the unknown portion of the mixture isn't too high. 

```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample* \
    --nbunknown 1 \
    --init SVD \
    --confidence 95 2500 \
    --outdir ci \
    --bedmethyl \
    --plot
```

You can only specify (in percent) the sample purity if you have it to make the estimation better.  It also makes the optimisation problem identifiable for the one sample, one known cell type case. 

```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample{1..10}.bed \
    --nbunknown 1 \
    --init SVD \
    --purity 60 80 90 20 50 90 100 30 50 10 \
    --outdir purity \
    --bedmethyl \
    --plot 
```

With the --confidence flag (arguments are confidence level in percentage and number of bootstrap iterations), you can obtain confidence intervals for the estimates and the --plot flag generates plots so that you can visualise the proportions estimates like this:


![proportions_stackedbar](https://github.com/user-attachments/assets/a3d0e144-d222-4595-8fe0-8548c9f1c992)
![proportions_bar_sample1](https://github.com/user-attachments/assets/f1e5f9dd-21c2-4a0a-b806-fa00481d4972)
