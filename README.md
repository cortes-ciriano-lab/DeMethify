<img width="402" alt="DeMethify" src="https://user-images.githubusercontent.com/79879340/220681790-e3a7edd0-d54c-4a49-b45a-95dca68c44b7.png">

                                       
DeMethify is a partial-reference based methylation deconvolution algorithm that uses a weighted constrained version of an iteratively optimized negative matrix factorization algorithm. 

## Flags and Arguments

| Option               | Description                                                                                                                    |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `--methfreq`         | Methylation frequency file path (values between 0 and 1).                                                                      |
| `--ref`              | Methylation reference matrix file path.                                                                                        |
| `--iterations`       | Numbers of iterations for outer and inner loops (default without purity = 10000, 20; with purity = 100, 500).                   |
| `--nbunknown`        | Number of unknown cell types to estimate.                                                                                      |
| `--purity`           | The purities of the samples in percent [0,100], if known.                                                                      |
| `--termination`      | Termination condition for cost function (default = 1e-2).                                                                      |
| `--init`             | Initialisation option, the default is `uniform_`, and the options are: `uniform`, `uniform_`, `beta`, `SVD`, `ICA`.            |
| `--outdir`           | Output directory.                                                                                                              |
| `--fillna`           | Replace every NA by 0 in the given data.                                                                                       |
| `--ic`               | Select number of unknown cell types by minimising a criterion (`AIC`, `BIC`, `CCC`, `BCV`, `minka`).                                     |
| `--confidence`       | Outputs bootstrap confidence intervals, takes confidence level and bootstrap iteration numbers as input.                        |
| `--plot`             | Plot cell type proportions estimates for each sample, eventually with confidence intervals.                                    |
| `--restart`          | Number of random restarts among which to select the one with the lowest cost/highest loglikelihood.                            |
| `--seed`             | Set a seed integer number for random number generation for reproducibility.                                                    |
| `--noprint`          | Does not show the logo.                                                                                                        |
| `--bedmethyl`        | Flag to indicate that the input will be bedmethyl files, modkit style.                                                         |





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


### Input format

The expected **reference** input format is either a **csv file** with methylation frequency values **between 0 and 1**, with rows corresponding to CpG sites and columns to cell types:


| Monocytes_EPIC | B-cells_EPIC | CD4T-cells_EPIC | NK-cells_EPIC | CD8T-cells_EPIC | Neutrophils_EPIC |
|----------------|--------------|-----------------|---------------|-----------------|------------------|
| 0.9484         | 0.9447       | 0.9438          | 0.9394        | 0.9527          | 0.9354           |
| 0.0426         | 0.0518       | 0.0425          | 0.0366        | 0.0398          | 0.0358           |
| 0.9235         | 0.8866       | 0.8822          | 0.9164        | 0.8956          | 0.9253           |
| 0.0499         | 0.0727       | 0.0468          | 0.0723        | 0.0536          | 0.0618           |
| 0.7557         | 0.8279       | 0.854           | 0.8148        | 0.8813          | 0.729            |
| 0.8648         | 0.865        | 0.8479          | 0.8596        | 0.8409          | 0.8644           |
| 0.0205         | 0.2352       | 0.0904          | 0.0307        | 0.2144          | 0.0122           |



or a **bedmethyl file** of the same kind, in this case you need to specify the `--bedmethyl` flag: 

| chrom  | start      | end        | Monocytes_EPIC | B-cells_EPIC | CD4T-cells_EPIC | NK-cells_EPIC | CD8T-cells_EPIC | Neutrophils_EPIC |
|--------|------------|------------|----------------|--------------|-----------------|---------------|-----------------|------------------|
| chr12  | 121416512  | 121416513  | 0.9484         | 0.9447       | 0.9438          | 0.9394        | 0.9527          | 0.9354           |
| chr1   | 6088550    | 6088551    | 0.0426         | 0.0518       | 0.0425          | 0.0366        | 0.0398          | 0.0358           |
| chr1   | 202092005  | 202092006  | 0.9235         | 0.8866       | 0.8822          | 0.9164        | 0.8956          | 0.9253           |
| chr17  | 29150100   | 29150101   | 0.0499         | 0.0727       | 0.0468          | 0.0723        | 0.0536          | 0.0618           |
| chr1   | 61549982   | 61549983   | 0.7557         | 0.8279       | 0.854           | 0.8148        | 0.8813          | 0.729            |
| chr3   | 195943366  | 195943367  | 0.8648         | 0.865        | 0.8479          | 0.8596        | 0.8409          | 0.8644           |
| chr14  | 38091470   | 38091471   | 0.0205         | 0.2352       | 0.0904          | 0.0307        | 0.2144          | 0.0122           |

In the same way, the expected sample format is either a csv file or a bedmethyl file where rows correspond to CpG sites, it should have one or two columns in whatever order corresponding to the methylation frequency "percent_modified" (required), and potentially total count "valid_coverage" (not required). There can be additional columns without changing anything. The methylation frequency for the bedmethyl files are expected to be percentages as they're usually given by tools like modkit, the methylation frequency for csv files are expected to be between 0 and 1 as usual:

| chrom  | start      | end        | valid_coverage | count_modified | percent_modified       |
|--------|------------|------------|----------------|----------------|-------------------------|
| chr1   | 227058070  | 227058071  | 55             | 4              | 7.2727272727272725      |
| chr1   | 3210424    | 3210425    | 52             | 46             | 88.46153846153845       |
| chr8   | 144649678  | 144649679  | 52             | 21             | 40.38461538461539       |
| chr1   | 3607425    | 3607426    | 41             | 31             | 75.60975609756098       |
| chr22  | 31317787   | 31317788   | 53             | 4              | 7.547169811320755       |
| chr20  | 42544716   | 42544717   | 50             | 0              | 0.0                     |
| chr2   | 72360187   | 72360188   | 49             | 47             | 95.91836734693877       |
| chr1   | 16164122   | 16164123   | 49             | 19             | 38.775510204081634      |

or:

| Valid Coverage | Percent Modified    |
|----------------|---------------------|
| 55             | 0.07272727272727273 |
| 52             | 0.8846153846153846  |
| 52             | 0.40384615384615385 |
| 41             | 0.7560975609756098  |
| 53             | 0.07547169811320755 |
| 50             | 0.0                 |
| 49             | 0.9591836734693877  |
| 49             | 0.3877551020408163  |


### Unsupervised case

If you've got no methylation reference matrix, you can still use DeMethify in a totally unsupervised fashion. Just leave out the --ref flag:

```
demethify \
    --methfreq output_gen/sample{1..10}.bed \
    --nbunknown 4 \
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
    --confidence 95 2500 \
    --outdir ci \
    --bedmethyl \
    --plot
```


![proportions_stackedbar](https://github.com/user-attachments/assets/a3d0e144-d222-4595-8fe0-8548c9f1c992)
![proportions_bar_sample1](https://github.com/user-attachments/assets/f1e5f9dd-21c2-4a0a-b806-fa00481d4972)

### Model selection

With the --ic flag, you can obtain the number of unknown cell types that minimises a set criterion, it can be corrected Bayesian Information Criterion with `BIC`, corrected Akaike Information Criterion with `AIC`, Brunet's Cophenetic Correlation Coefficient method with `CCC`, an adapted version of Owen and Parry's bi-cross-validation method with `BCV`, or an adapted version of the Minka-PCA method with `minka`. One can specify the number of restarts/number of folds for `CCC` and `BCV` by adding an int number argument after the method as in `--ic BCV 30`:


```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample{1..10}.bed \
    --bedmethyl \
    --ic AIC \
    --outdir bloblo \
    --plot
```

![ic_plot](https://github.com/user-attachments/assets/242a736f-18d3-4fe3-a908-abff294f07ab)

```
demethify \
    --ref output_gen/ref_matrix.bed \
    --methfreq output_gen/sample{1..10}.bed \
    --bedmethyl \
    --ic CCC 20 \
    --outdir bloblo \
    --plot
```

![ic_plot](https://github.com/user-attachments/assets/6416daa0-4af6-4fab-9816-7c2b98d820b7)



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


$n_s \geq 1$

Which means that the purity information makes the one sample estimation problem identifiable in the partial-reference based case with a single unknown cell type and a single known cell type. 


