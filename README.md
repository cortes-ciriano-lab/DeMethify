<img width="402" alt="DeMethify" src="https://user-images.githubusercontent.com/79879340/220681790-e3a7edd0-d54c-4a49-b45a-95dca68c44b7.png">

                                       
DeMethify is a partial-reference based methylation deconvolution algorithm that uses a weighted constrained version of an iteratively optimized negative matrix factorization algorithm. 

## Run DeMethify

After installing, you can run DeMethify with the following arguments in the case of read format data input:
```
python demethify --methfreq <methfreq_csv> --counts <counts_csv> --ref <ref_csv> --outdir <outdir> --nbunknown <nb_unknown>
```
in the case of no read format data input :

```
python demethify --methfreq <methfreq_csv> --noreadformat --ref <ref_csv> --outdir <outdir> --nbunknown <nb_unknown>
```

### Mandatory Arguments
Argument|Description
---|---
methfreq|Methylation frequency CSV file (values between 0 and 1)
ref|Reference methylation matrix CSV file
outdir|Output directory (can exist but must be empty)
nbunknown|Number of unknown cell types to estimate 

### Optional Arguments
Argument|Description
---|---
counts|Read counts CSV file
noreadformat|Flag to use when the data isn't using the read format (like Illumina epic arrays)
iterations|Numbers of iterations for outer and inner loops (default = 50000, 50)
termination|Termination condition for cost function (default = 1e-2)
