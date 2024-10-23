import numpy as np
import pandas as pd
import colorcet as cc  
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_proportions(df, ci_df, outdir):

    unique_ct = list(df.index)

    colors = sns.color_palette(cc.glasbey, len(unique_ct))

    color_mapping = {barcode: color for barcode, color in zip(unique_ct, colors)}
    clrs = [color_mapping[barcode] for barcode in unique_ct]
    plt.figure(figsize=(12, 8))
    ax = df.T.plot(kind='bar', stacked=True, figsize=(10, 6), color=clrs)

    plt.title('Proportion of Cell Types in Each Sample')
    plt.ylabel('Proportion')
    plt.xlabel('Samples')

    plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    outdir_plots = outdir + '/plots'
    if not os.path.exists(outdir_plots):
        os.mkdir(outdir_plots)
    plt.savefig(outdir_plots + '/proportions_stackedbar.png', dpi=300, bbox_inches='tight')

    sns.set(style="whitegrid")

    for sample in df.columns: 
        plt.figure(figsize=(12, 8))
        if(not ci_df.empty):
            ax = sns.barplot(x=df.index, y=df[sample], palette=clrs, ci=None)

            ci_values = ci_df[sample].apply(lambda x: (x[0], x[1])) 

            lower_bounds = np.array([ci[0] for ci in ci_values])
            upper_bounds = np.array([ci[1] for ci in ci_values])

            lower_error = abs(df[sample].values - lower_bounds)  
            upper_error = abs(upper_bounds - df[sample].values)  

            ax.errorbar(x=np.arange(len(df.index)), 
                        y=df[sample], 
                        yerr=[lower_error, upper_error],  
                        fmt='none',  
                        ecolor='black',  
                        capsize=5,  
                        capthick=2)  
        else:
            sns.barplot(x=df.index, y=df[sample], palette=clrs)

        plt.xlabel('Cell Types')
        plt.ylabel('Proportion')
        plt.title(f'Proportion of Cell Types in {sample}')
    
        plt.xticks(rotation=90)
    
        plt.savefig(outdir_plots + '/proportions_bar_' + sample[:-4] + '.png', dpi=300, bbox_inches='tight')

    print("Plots generated in " + outdir_plots)