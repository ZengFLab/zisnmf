import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scanpy.plotting._utils import savefig_or_show



def plot_rank_genes_groups(
    adata,
    groups = None,
    n_genes = 20,
    gene_symbols = None,
    key = "rank_genes_groups",
    fontsize = 8,
    ncols = 4,
    sharey = True,
    show = None,
    save = None,
    ax = None,
    **kwds,
):
    """\
    Plot ranking of genes.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groups
        The groups for which to show the gene ranking.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to
        use `.var_names`.
    n_genes
        Number of genes to show.
    fontsize
        Fontsize for gene names.
    ncols
        Number of panels shown per row.
    sharey
        Controls if the y-axis of each panels should be shared. But passing
        `sharey=False`, each panel has its own y-axis range.


    Examples
    --------

    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        sc.pl.rank_genes_groups(adata)


    Plot top 10 genes (default 20 genes)

    .. plot::
        :context: close-figs

        sc.pl.rank_genes_groups(adata, n_genes=10)

    .. currentmodule:: scanpy

    See also
    --------
    tl.rank_genes_groups

    """
    if "n_panels_per_row" in kwds:
        n_panels_per_row = kwds["n_panels_per_row"]
    else:
        n_panels_per_row = ncols
    if n_genes < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_genes has not been implemented for "
            f"this plot. Received n_genes={n_genes}."
        )

    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    from matplotlib import gridspec

    fig = None
    if ax is None:
        fig = plt.figure(
            figsize=(
                n_panels_x * rcParams["figure.figsize"][0],
                n_panels_y * rcParams["figure.figsize"][1],
            )
        )
    
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ax1 = ax
    ymin = np.inf
    ymax = -np.inf
    for count, group_name in enumerate(group_names):
        gene_names = adata.uns[key]["names"][group_name][:n_genes]
        scores = adata.uns[key]["scores"][group_name][:n_genes]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(scores))
            ymax = max(ymax, np.max(scores))

            if ax0 is None:
                if ax is None:
                    ax1 = fig.add_subplot(gs[count])
                ax0 = ax1
            else:
                if ax is None:
                    ax1 = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(scores)
            ymax = np.max(scores)
            ymax += 0.3 * (ymax - ymin)

            if ax is None:
                ax1 = fig.add_subplot(gs[count])
            ax1.set_ylim(ymin, ymax)

        ax1.set_xlim(-0.9, n_genes - 0.1)

        # Mapping to gene_symbols
        if gene_symbols is not None:
            if adata.raw is not None and adata.uns[key]["params"]["use_raw"]:
                gene_names = adata.raw.var[gene_symbols][gene_names]
            else:
                gene_names = adata.var[gene_symbols][gene_names]

        # Making labels
        for ig, gene_name in enumerate(gene_names):
            ax1.text(
                ig,
                scores[ig],
                gene_name,
                rotation="vertical",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=fontsize,
            )

        ax1.set_title(f"{group_name}")
        if count >= n_panels_x * (n_panels_y - 1):
            ax1.set_xlabel("Ranking")

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax1.set_ylabel("Relevance score")

    if True:
    #if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ymin -= 0.3 * (ymax - ymin)
        ax1.set_ylim(ymin, ymax)

    ax = ax1

    writekey = f"rank_genes_groups_{adata.uns[key]['params']['groupby']}"
    savefig_or_show(writekey, show=show, save=save)



def add_V_components_to_anndata(adata, V_comp, cluster_names, var_names, key_added='zisnmf2', groupby='cluster',
                                use_raw=True, layer='X'):
    adata.uns[key_added] = {}
    rank_stats = None
    for i, group_i in enumerate(cluster_names):
        if rank_stats is None:
            idx = pd.MultiIndex.from_tuples([(group_i,'names')])
            rank_stats = pd.DataFrame(columns=idx)

        global_indices = np.argsort(-V_comp[i])
        rank_stats[group_i, 'names'] = var_names[global_indices]
        rank_stats[group_i, 'scores'] = V_comp[i][global_indices]
        rank_stats[group_i, 'logfoldchanges'] = V_comp[i][global_indices]
        rank_stats[group_i, 'pvals'] = np.zeros_like(V_comp[i][global_indices])
        rank_stats[group_i, 'pvals_adj'] = np.zeros_like(V_comp[i][global_indices])

    ##
    adata.uns[key_added]['params'] = dict(
    groupby=groupby,
    reference='rest',
    groups=cluster_names,
    method='zisnmf2',
    use_raw=use_raw,
    layer=layer,
    )

    dtypes = {
        'names': 'O',
        'scores': 'float32',
        'logfoldchanges': 'float32',
        'pvals': 'float32',
        'pvals_adj': 'float32'
        }
    rank_stats.columns = rank_stats.columns.swaplevel()
    for col in rank_stats.columns.levels[0]:
        adata.uns[key_added][col]=rank_stats[col].to_records(
    index=False, column_dtypes=dtypes[col]
    )
        
    return adata


# Function to format y-axis labels with two decimal places
def format_func(value, tick_number=None):
    return f"{value:.2f}"

def plot_genes_across_groups(df, num_columns, sharey=False, save_to=None, dpi=300):
    """
    Plots each column of the given DataFrame in a grid of subplots.
    Parameters:
    - df: pd.DataFrame containing the data to plot.
    - num_columns: int, number of columns in the subplot grid.

    Returns:
    - None, displays the plots.
    """
    # Number of columns and subplots configuration
    num_plots = df.shape[1]
    num_rows = (num_plots + num_columns - 1) // num_columns  # Calculate number of rows needed

    # Create subplots with constrained_layout for better alignment
    fig, axes = plt.subplots(num_rows, num_columns, 
                             figsize=(num_columns * rcParams["figure.figsize"][0], 
                                      num_rows * rcParams["figure.figsize"][1]), 
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes

    # Determine the global y limits
    global_y_min = df.min().min()
    global_y_max = df.max().max()

    # Plot each column in a separate subplot
    for idx, col in enumerate(df.columns):
        ax = axes[idx]
        values = df[col]
        bars = ax.bar(df.index, values, color=sns.color_palette("viridis", len(values)))
        ax.set_ylabel(col)
        if sharey:
            ax.set_ylim(global_y_min, global_y_max)  # Set consistent y limits if desired

        # Format y-axis labels
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

        # Color bars according to their values
        for bar, value in zip(bars, values):
            bar.set_color(sns.color_palette("viridis", as_cmap=True)(value))

    # Set x-axis labels for the bottom subplots only
    for ax in axes[-num_columns:]:
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(df.index, rotation=90)

    # Remove any unused subplots
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Save the plot if the save_to parameter is provided
    if save_to:
        plt.savefig(save_to, dpi=dpi)
        print(f"Plot saved to {save_to} with DPI of {dpi}")

    # Adjust layout with constrained_layout
    plt.show()

    


def plot_genes_between_groups(df, left_col, right_col, xlabel='', ylabel='', title='', save_to=None, dpi=300, cmap='Reds', swap_axes=False):
    if swap_axes:
        _plot_genes_between_groups1(df, left_col, right_col, xlabel, ylabel, title, save_to, dpi, cmap)
    else:
        _plot_genes_between_groups2(df, left_col, right_col, ylabel, xlabel, title, save_to, dpi, cmap)

def _plot_genes_between_groups1(df, left_col, right_col, xlabel='', ylabel='', title='', save_to=None, dpi=300, cmap='Reds'):
    """
    Create a back-to-back bar plot from a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    left_col (str): Name of the column for the left-side bars.
    right_col (str): Name of the column for the right-side bars.
    xlabel (str): Label for the x-axis. Default is ''.
    ylabel (str): Label for the y-axis. Default is ''.
    title (str): Title of the plot. Default is ''.
    save_to (str): File path to save the plot. If None, the plot will not be saved.
    dpi (int): Resolution of the saved figure in dots per inch. Default is 300.
    """
    # Set the figure and axes
    fig, ax = plt.subplots()

    # Normalize values for colormap
    left_values = df[left_col].values
    right_values = df[right_col].values

    min_value = df.values.ravel().min()
    max_value = df.values.ravel().max()

    norm_left = (left_values - min_value) / (max_value - min_value)
    norm_right = (right_values - min_value) / (max_value - min_value)

    # Create colormap using viridis
    viridis_cmap = plt.get_cmap(cmap)

    # Get colors based on normalized values
    left_colors = viridis_cmap(norm_left)
    right_colors = viridis_cmap(norm_right)

    # Plotting the left column as negative values for left bars
    ax.barh(df.index, -df[left_col], color=left_colors, label=left_col)

    # Plotting the right column as positive values for right bars
    ax.barh(df.index, df[right_col], color=right_colors, label=right_col)

    # Set labels and title based on function parameters
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8)  # Add a vertical line at x=0

    # Set y-ticks to object names
    ax.set_yticks(df.index)

    # Add vertical grid lines
    ax.xaxis.grid(True, which='both', color='gray', linestyle='--', linewidth=0.7)
        
    # Add the left_col label and right_col label under the x-axis
    ax.text( - 0.2, len(df.index) + 0.2, left_col, color='black', ha='right', va='center', fontsize=12)
    ax.text( 0.2, len(df.index) + 0.2, right_col, color='black', ha='left', va='center', fontsize=12)

    # Add legend
    #ax.legend()

    # Save the plot if the save_to parameter is provided
    if save_to:
        plt.savefig(save_to, dpi=dpi)
        print(f"Plot saved to {save_to} with DPI of {dpi}")

    # Show the plot
    plt.show()

    




def _plot_genes_between_groups2(df, top_col, bottom_col, xlabel='', ylabel='', title='', save_to=None, dpi=300, cmap='Reds'):
    """
    Create a top-down back-to-back bar plot from a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    top_col (str): Name of the column for the top bars.
    bottom_col (str): Name of the column for the bottom bars.
    xlabel (str): Label for the x-axis. Default is 'Response'.
    title (str): Title of the plot. Default is 'Top-Down Back-to-Back Bar Plot of Drug Responses'.
    save_to (str): File path to save the plot. If None, the plot will not be saved.
    dpi (int): Resolution of the saved figure in dots per inch. Default is 300.
    """
    # Set the figure and axes
    fig, ax = plt.subplots()

    # Normalize values for colormap
    top_values = df[top_col].values
    bottom_values = df[bottom_col].values

    min_value = df.values.ravel().min()
    max_value = df.values.ravel().max()

    norm_top = (top_values - min_value) / (max_value - min_value)
    norm_bottom = (bottom_values - min_value) / (max_value - min_value)

    # Create colormap using viridis
    viridis_cmap = plt.get_cmap(cmap)

    # Get colors based on normalized values
    top_colors = viridis_cmap(norm_top)
    bottom_colors = viridis_cmap(norm_bottom)

    # Plotting the top column as positive values for top bars
    ax.bar(df.index, top_values, color=top_colors, label=top_col)

    # Plotting the bottom column as negative values for bottom bars
    ax.bar(df.index, -bottom_values, color=bottom_colors, label=bottom_col)

    # Set labels and title based on function parameters
    ax.set_ylabel(xlabel, fontsize=12)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0

    # Set y-ticks to display the positive and negative values appropriately
    y_ticks_bottom = np.arange(0, bottom_values.max()+1, 0.2)
    y_ticks_top = np.arange(0, top_values.max()+1, 0.2)
    y_ticks = np.hstack([-y_ticks_bottom, y_ticks_top])
    #y_ticks = np.arange(-bottom_values.max() - 1, top_values.max() + 1, 0.2)
    ax.set_yticks(y_ticks)

    ax.set_ylim(-bottom_values.max() - 0.2, top_values.max() + 0.2)
    ax.set_xlim(-2, len(df.index.unique()) + 1)

    # Set y-tick labels to show absolute values of negative ticks
    y_tick_labels = [format_func(abs(y)) for y in y_ticks]
    ax.set_yticklabels(y_tick_labels)

    # Add horizontal grid lines
    ax.yaxis.grid(True, which='both', color='gray', linestyle='--', linewidth=0.7)

    # Add labels for top and bottom bars
    ax.text(len(df.index.unique()) / 2, top_values.max(), top_col, color='black', ha='center', va='bottom', fontsize=12)
    ax.text(len(df.index.unique()) / 2, -bottom_values.max(), bottom_col, color='black', ha='center', va='top', fontsize=12)

    # Add legend
    #ax.legend()

    # Rotate x-tick labels
    plt.xticks(rotation=90)

    # Save the plot if the save_to parameter is provided
    if save_to:
        plt.savefig(save_to, dpi=dpi)
        print(f"Plot saved to {save_to} with DPI of {dpi}")
        
    # Show the plot
    plt.show()

    

