<img width="402" alt="DeMethify" src="https://user-images.githubusercontent.com/79879340/220681790-e3a7edd0-d54c-4a49-b45a-95dca68c44b7.png">

                                       
DeMethify is a partial-reference based methylation deconvolution algorithm that uses a weighted constrained version of an iteratively optimized negative matrix factorization algorithm. 

## Flags and Arguments

| Option           | Description                                                                                                                     |
|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `--methfreq`     | Methylation frequency file path (values between 0 and 1)                                                                        |
| `--ref`          | Methylation reference matrix file path                                                                                          |
| `--iterations`   | Numbers of iterations for outer and inner loops (default without purity = 10000, 20; with purity = 100, 500)                     |
| `--nbunknown`    | Number of unknown cell types to estimate                                                                                        |
| `--purity`       | The purities of the samples in percent [0,100], if known                                                                        |
| `--termination`  | Termination condition for cost function (default = 1e-2)                                                                        |
| `--init`         | Initialisation option (default = random uniform)                                                                                |
| `--outdir`       | Output directory                                                                                                                |
| `--fillna`       | Replace every NA by 0 in the given data                                                                                         |
| `--ic`           | Select number of unknown cell types by minimising an information criterion (AIC or BIC)                                         |
| `--confidence`   | Outputs bootstrap confidence intervals, takes confidence level and bootstrap iteration numbers as input                         |
| `--plot`         | Plot cell type proportions estimates for each sample, eventually with confidence intervals.                                     |
| `--restart`      | Number of random restarts among which to select the one with the lowest cost/highest loglikelihood                              |
| `--seed`         | Set a seed integer number for random number generation for reproducibility.                                                     |
| `--noprint`      | Does not show the logo.                                                                                                         |
| `--bedmethyl`    | Flag to indicate that the input will be bedmethyl files, modkit style                                                           |
| `--counts`       | Read counts file path                                                                                                           |
| `--noreadformat` | Flag to use when the data isn't using the read format (e.g., Illumina epic arrays)                                              |



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

Here is a flowchart to run you through the different use cases for DeMethify. 

![flowchart-fun](https://github.com/user-attachments/assets/eb174261-8852-4436-aaf7-13511f0fbdfe)


### Unsupervised case

If you've got no methylation reference matrix, you can still use DeMethify in a totally unsupervised fashion. Just leave out the --ref flag:

```
demethify \
    --methfreq output_gen/sample{1..10}.bed \
    --nbunknown 4 \
    --init SVD \
    --outdir unsupervised \
    --bedmethyl \
    --plot
```

### Reference based case

If you want to perform fully reference-based methylation deconvolution, just leave out the --nbunknown flag:

```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample{1..10}.bed \
    --bedmethyl \
    --outdir output_ref_based \
    --plot
```

### Partial-reference based case

If you've got a number of samples greater or equal than 2, you can use the partial-reference based algorithm to jointly estimate the unknown cell type portion methylation profile and the proportions of all known and unknown cell types, otherwise you can use the reference based algorithm (if you don't specify --nbunknown) and hope that the unknown portion of the mixture isn't too high. 

```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample{1..10}.bed \
    --nbunknown 1 \
    --init SVD \
    --confidence 95 2500 \
    --outdir ci \
    --bedmethyl \
    --plot
```

### Partial-reference based case with purity

You can specify (in percent) the sample purity if you have it to make the estimation better.  It also makes the optimisation problem identifiable for the one sample, one known cell type case. 

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

### Confidence intervals

With the --confidence flag (arguments are confidence level in percentage and number of bootstrap iterations), you can obtain confidence intervals for the estimates and the --plot flag generates plots so that you can visualise the proportions estimates like this:

```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample{1..10}.bed \
    --nbunknown 1 \
    --init SVD \
    --confidence 95 2500 \
    --outdir ci \
    --bedmethyl \
    --plot
```


![proportions_stackedbar](https://github.com/user-attachments/assets/a3d0e144-d222-4595-8fe0-8548c9f1c992)
![proportions_bar_sample1](https://github.com/user-attachments/assets/f1e5f9dd-21c2-4a0a-b806-fa00481d4972)

## Identifiability of the estimation
$n_s$ : number of samples
$n_u$ : number of unknown cell types to estimate
$n_c$ : number of known cell types
$n_{cpg}$ : number of CpG sites

In the partial-reference based case without purity, the estimation problem is identifiable when:

$n_s \geq \frac{n_u n_{cpg}}{n_{cpg} - n_u - n_c + 1}$

When $n_u = 1$, we have:

$n_s \geq \frac{n_{cpg}}{n_{cpg} - n_c}$

The ratio on the right is in $(1,2]$ for most real-life situations, which means that for the estimation problem to be identifiable in the partial-reference based case without purity with a single unknown cell type we need at least 2 samples. 

In the partial-reference based case with purity, the estimation problem is identifiable when:

$n_s \geq \frac{n_u n_{cpg}}{n_{cpg} - n_u - n_c + 2}$

When $n_u = 1$, we have:

$n_s \geq \frac{n_{cpg}}{n_{cpg} - n_c + 1}$

And with $n_c = 1$, we have:

$n_s \geq 1$

Which means that the purity information makes the one sample estimation problem identifiable in the partial-reference based case with a single unknown cell type and a single known cell type. 


