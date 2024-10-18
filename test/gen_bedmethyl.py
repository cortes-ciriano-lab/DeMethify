import os
import numpy as np
import numpy.random as rd

def gen_param_u(R_full, read_depth, trunc, unknown, nb_samples, disp=1.0):
    nb_cpg, nb_celltypes = R_full.shape

    alpha_sim_1 = rd.dirichlet(np.ones(trunc), nb_samples).T
    alpha_sim_2 = rd.dirichlet(np.ones(nb_celltypes - trunc), 1).T
    alpha_sim = np.concatenate([alpha_sim_1 * (1 - unknown), alpha_sim_2 * unknown])

    d_x = rd.poisson(read_depth, (nb_cpg, nb_samples))
    
    R_full = R_full + ((R_full == 0) * 1e-10) - ((R_full == 1) * 1e-10)
    R_full = rd.beta(disp * R_full, disp * (1 - R_full))
    beta_sim = R_full @ alpha_sim
    x = rd.binomial(d_x, beta_sim)
    m_u = R_full[:, trunc:] @ alpha_sim_2 
    
    return x, d_x, np.concatenate([alpha_sim[:trunc,:], unknown_portion]), m_u

def gen_param(R_full, read_depth, nb_samples, disp=1.0):
    nb_cpg, nb_celltypes = R_full.shape

    alpha_sim = rd.dirichlet(np.ones(nb_celltypes), nb_samples).T

    d_x = rd.poisson(read_depth, (nb_cpg, nb_samples))
    
    R_full = R_full + ((R_full == 0) * 1e-16) - ((R_full == 1) * 1e-16)
    R_full = rd.beta(disp * R_full, disp * (1 - R_full))
    beta_sim = R_full @ alpha_sim

    x = rd.binomial(d_x, beta_sim)
    
    return x, d_x, alpha_sim

ref_file = "bed1.bed"
ref = pd.read_csv(ref_file, sep='\t')
random_subsample = 5000
ref = ref.sample(n=random_subsample)
pos, df = ref.iloc[:,:3], ref.iloc[:,3:]
read_depth = 50
nb_known_cell_types = 5
gen_u = "select"
select_cell_types = ['Adipocytes', 'Cortical_neurons', 'Hepatocytes', 'Lung_cells', 'Pancreatic_beta_cells'] 
unknown_portion = np.reshape(np.array([0.4, 0.2, 0.1]), (1,3))
nb_samples = 3
outdir = "output_gen"


output_folder = os.path.join(os.getcwd(), outdir)
if not os.path.exists(output_folder):
    print(f'Creating directory {output_folder} to store results')
    os.mkdir(output_folder)

if gen_u:
    if gen_u == "random":
        known_cell_types = random_column_names = list(rd.choice(df.columns, nb_known_cell_types, replace=False))
    elif gen_u == "first":
        known_cell_types = list(df.columns)[:nb_known_cell_types]
    elif gen_u == "select":
        known_cell_types = select_cell_types
        
    df = df[known_cell_types + [col for col in df.columns if col not in known_cell_types]]
    
    meth_counts, counts, alpha_sim, meth_u = gen_param_u(df.values, read_depth, nb_known_cell_types, unknown_portion, nb_samples)   

else:
    known_cell_types = df.columns
    meth_counts, counts, alpha_sim = gen_param(df.values, read_depth, nb_samples)

alpha_sim_df = pd.DataFrame(alpha_sim)
index_name = known_cell_types
if gen_u:
    index_name += ["unknown_cell_1"]
    meth_u_df = pd.DataFrame(meth_u)
    meth_u_df.columns = ["unknown_cell_1"]
    meth_u_df.to_csv(output_folder + '/meth_profile_sim.csv',  sep='\t', index = False)
alpha_sim_df.index = index_name
alpha_sim_df.columns = ["sample" + str(i + 1) for i in range(nb_samples)]
alpha_sim_df.to_csv(output_folder + '/proportions_sim.csv',  sep='\t', index = True)

for i in range(nb_samples):
    sample = pos.copy()
    sample['valid_coverage'] = counts[:,i:i+1]
    sample['count_modified'] = meth_counts[:,i:i+1]
    sample['percent_modified'] =  (sample['count_modified'] / sample['valid_coverage']) * 100

    sample.to_csv(output_folder + '/sample' + str(i + 1) + '.bed',  sep='\t', index = False)
