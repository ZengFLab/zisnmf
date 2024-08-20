import numpy as np
import pandas as pd
from scipy.stats import nbinom
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

def generate_zinb_counts(mean, theta, size, dropout_rate):
    p = theta / (theta + mean)
    counts = nbinom.rvs(theta, p, size=size)
    if isinstance(size, tuple):
        dropout_mask = np.random.rand(*size) < dropout_rate
    else:
        dropout_mask = np.random.rand(size) < dropout_rate
    counts[dropout_mask] = 0
    return counts

def simulate_discrete_cell_populations(n_clusters=5, n_cells_per_cluster=1000, n_genes=20000, 
                                       n_markers_per_cluster=50, expression_mean=5, expression_scale=0.8, theta=10,
                                       gain=5, de_loc=None, de_scale=0.5, dropout_rate=0.9, de_dropout_rate=0.1,
                                       n_housekeeping_genes=200, housekeeping_mean_expression=10):
    n_cells = n_clusters * n_cells_per_cluster
    de_loc = np.log(np.power(10, gain/10)) if de_loc is None else de_loc

    genes = [f'Gene{i}' for i in range(n_genes)]
    clusters = []
    degs_dict = {}

    # 选择housekeeping genes
    housekeeping_genes = np.random.choice(n_genes, n_housekeeping_genes, replace=False)
    non_housekeeping_genes = np.setdiff1d(np.arange(n_genes), housekeeping_genes)

    for cluster_id in range(n_clusters):
        base_mean_expression = np.random.lognormal(mean=expression_mean, sigma=expression_scale, size=(n_cells_per_cluster, n_genes))
        
        # 为housekeeping genes设置更高的平均表达量
        base_mean_expression[:, housekeeping_genes] = np.random.lognormal(mean=housekeeping_mean_expression, sigma=expression_scale, size=(n_cells_per_cluster, n_housekeeping_genes))
        
        base_counts = generate_zinb_counts(base_mean_expression, theta, (n_cells_per_cluster, n_genes), dropout_rate)

        degs = np.random.choice(non_housekeeping_genes, n_markers_per_cluster, replace=False)
        degs_dict[f'Cluster{cluster_id + 1}'] = [genes[deg] for deg in degs]

        for deg in degs:
            deg_mean = np.random.lognormal(mean=expression_mean+de_loc, sigma=de_scale, size=base_mean_expression[:, deg].shape)
            deg_counts = generate_zinb_counts(deg_mean[:, np.newaxis], theta, (n_cells_per_cluster, 1), de_dropout_rate).flatten()
            base_counts[:, deg] = deg_counts
        
        cluster_df = pd.DataFrame(base_counts, columns=genes)
        cluster_df['Cluster'] = f'Cluster{cluster_id + 1}'
        clusters.append(cluster_df)
    
    data = pd.concat(clusters)
    adata = sc.AnnData(data.iloc[:, :-1].values)
    adata.obs['Cluster'] = data['Cluster'].values
    adata.var_names = genes

    simulate_results = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'n_clusters': n_clusters,
        'n_markers_per_cluster': n_markers_per_cluster,
        'adata': adata,
        'degs_dict': degs_dict,
        'housekeeping_genes': [genes[i] for i in housekeeping_genes],
    }
    
    return simulate_results

## Example usage:
#results = simulate_discrete_cell_populations()
#
## Save the AnnData object for further analysis
#results['adata'].write('synthetic_single_cell_data_with_dropout.h5ad')
#
## Save the DEGs information
#degs_df = pd.DataFrame.from_dict(results['degs_dict'], orient='index').transpose()
#degs_df.to_csv('differentially_expressed_genes.csv', index=False)
#
#import pickle
#
#with open('tree_simulate_results.pkl', 'wb') as f:
#    pickle.dump(results, f)







def generate_marker_genes(num_genes, num_cell_types, num_cell_subtypes, num_marker_genes, shared_ratio=0.2):
    genes = range(num_genes)
    used_markers = set()
    markers = {}
    for i in range(num_cell_types):
        remaining_genes = np.setdiff1d(genes, list(used_markers))
        cell_type_markers = np.random.choice(remaining_genes, num_marker_genes, replace=False)
        markers[f"CellType_{i}"] = cell_type_markers
        
        for j in range(num_cell_subtypes - 1):
            shared_genes = np.random.choice(cell_type_markers, int(shared_ratio * num_marker_genes), replace=False)
            unique_genes = np.random.choice(remaining_genes, num_marker_genes - len(shared_genes), replace=False)
            markers[f"CellType_{i}_SubType_{j}"] = np.concatenate([shared_genes, unique_genes])
        
        markers[f"CellType_{i}_SubType_{num_cell_subtypes - 1}"] = cell_type_markers
        used_markers.update(cell_type_markers)
    return markers

def generate_single_cell_data(num_cells, num_genes, markers, num_cell_types, num_cell_subtypes, mean_expression=5, theta=10, de_loc=1, de_scale=0.1, dropout_rate=0.3):
    base_mean_expression = np.random.lognormal(mean=mean_expression, sigma=0.5, size=(num_cells, num_genes))
    data = generate_zinb_counts(base_mean_expression, theta, (num_cells, num_genes), dropout_rate)
    
    cell_types = []
    cell_subtypes = []

    for i in range(num_cells):
        cell_type = np.random.choice(range(num_cell_types))
        cell_subtype = np.random.choice(range(num_cell_subtypes))
        cell_types.append(f"CellType_{cell_type}")
        cell_subtypes.append(f"CellType_{cell_type}_SubType_{cell_subtype}")

        marker_genes = markers[f"CellType_{cell_type}_SubType_{cell_subtype}"]
        for gene in marker_genes:
            gene_mean = np.random.lognormal(mean=mean_expression + de_loc, sigma=de_scale, size=1)
            gene_expression = generate_zinb_counts(gene_mean, theta, 1, dropout_rate)
            data[i, gene] = gene_expression

    return data, cell_types, cell_subtypes

def simulate_hierarchical_cell_populations(num_genes=2000, num_cell_types=5, num_cell_subtypes=3, num_marker_genes=50, shared_ratio=0.2, 
                                           num_cells=5000, mean_expression=5, theta=10, gain=5, de_loc=None, de_scale=0.1, dropout_rate=0.3):
    de_loc = np.log(np.power(10, gain/10)) if de_loc is None else de_loc
    markers = generate_marker_genes(num_genes, num_cell_types, num_cell_subtypes, num_marker_genes, shared_ratio)
    data, cell_types, cell_subtypes = generate_single_cell_data(num_cells, num_genes, markers, num_cell_types, num_cell_subtypes, mean_expression, theta, de_loc, de_scale, dropout_rate)
    
    adata = sc.AnnData(data)
    adata.obs['cell_type'] = pd.Categorical(cell_types)
    adata.obs['cell_subtype'] = pd.Categorical(cell_subtypes)
    
    simulate_results = {
        'n_cells': num_cells,
        'n_genes': num_genes,
        'n_clusters': num_cell_types*num_cell_subtypes,
        'n_cell_types': num_cell_types,
        'n_cell_subtypes': num_cell_subtypes,
        'n_markers_per_cluster': num_marker_genes,
        'adata': adata,
        'degs_dict': markers
    }
    
    return simulate_results

## Example usage:
#results = simulate_hierarchical_cell_populations()
#
## Save the AnnData object for further analysis
#results['adata'].write('synthetic_single_cell_data_with_hierarchy.h5ad')
#
## Save the markers information
#markers_df = pd.DataFrame.from_dict(results['markers'], orient='index').transpose()
#markers_df.to_csv('differentially_expressed_genes_hierarchy.csv', index=False)
#
#import pickle
#
#with open('hierarchical_simulate_results.pkl', 'wb') as f:
#    pickle.dump(results, f)