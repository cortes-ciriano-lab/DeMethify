import numpy as np
import pandas as pd
import colorcet as cc  
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Generates plots to visualize the estimated proportion vectors and evaluate the performance of various model selection criteria.
def plot_proportions(df, ci_df, outdir, list_ic=None):

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

    if list_ic:
        
        plt.figure(figsize=(8, 6))
        x_values = [i + 1 for i in range(len(list_ic))]
        plt.plot(x_values, list_ic, marker='x', linestyle='-', linewidth=1.5, markersize=8, markeredgecolor='red', label='IC Curve')
        plt.xlabel("Number of Unknown Components", fontsize=12)
        plt.ylabel("IC Values", fontsize=12)
        plt.title("IC vs. Number of Components", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()


        min_ic_index = np.argmin(list_ic)  
        min_ic_value = list_ic[min_ic_index]  
        min_ic_components = x_values[min_ic_index]  
        
        annotation_x = min_ic_components
        annotation_y = min_ic_value
        
        plt.text(0.05, 0.95, 
         f"Min IC at {min_ic_components}", 
         color='red', fontsize=10, 
         transform=plt.gca().transAxes, 
         verticalalignment='top', 
         horizontalalignment='left')

        plt.savefig(outdir_plots + '/ic_plot.png', dpi=300, bbox_inches='tight')
    

    print("Plots generated in " + outdir_plots)