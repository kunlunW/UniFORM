"""
EDA to determine variations in histograms

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 12/02/2024

Dependencies:
    - numpy
    - skimage
    - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from tqdm import tqdm
import anndata as ad

def preprocess_raw(image):
    ch = np.log10(image + 1)
    return ch

def plot_global_gmm(X):
    X = X.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, init_params='random_from_data', max_iter=1000, random_state=10, covariance_type='full')
    gmm.fit(X)
    
    predictions = gmm.predict(X)
    class1 = X[predictions == 1]
    class0 = X[predictions == 0]
    
    if not class0.size:
        class0 = np.array([0])  # Fallback value
    if not class1.size:
        class1 = np.array([0])  # Fallback value
    neg, pos = (class1, class0) if max(class0) > max(class1) else (class0, class1)
    cut = max(neg)
    return cut


def plot_local_gmm(X, bin_counts, global_min, global_max):
    X = X.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, init_params='random_from_data', max_iter=1000, random_state=10, covariance_type='full')
    gmm.fit(X)
    
    predictions = gmm.predict(X)
    class1 = X[predictions == 1]
    class0 = X[predictions == 0]
    
    if not class0.size:
        class0 = np.array([0])  # Fallback value
    if not class1.size:
        class1 = np.array([0])  # Fallback value
    
    neg, pos = (class1, class0) if max(class0) > max(class1) else (class0, class1)
    x_axis = np.linspace(global_min, global_max, bin_counts)
    
    y_axis0 = norm.pdf(x_axis, gmm.means_[0], np.sqrt(gmm.covariances_[0])) * gmm.weights_[0]
    y_axis1 = norm.pdf(x_axis, gmm.means_[1], np.sqrt(gmm.covariances_[1])) * gmm.weights_[1]
    
    cut = max(neg)
    return cut, x_axis, y_axis0, y_axis1


def uniform_calculate_histogram(adata, marker_to_plot, rep="X", bin_counts=1024, subplots_per_row=4, dpi=600,
                    xlims=None, ylims=None, save_filename=None, colormap='tab10',
                    plot_local_threshold=True, plot_global_threshold=True, plot_legend=True, 
                    transparent_background=True, plot_width=3):
    
    # Extract sample names and marker names
    sample_names = adata.obs['sample_id'].unique()
    marker_list = adata.var['marker_name'].tolist()
    num_samples = len(sample_names)

    if rep == "X":
        data_matrix = adata.X
    elif rep == "normalized" and "normalized" in adata.layers:
        data_matrix = adata.layers["normalized"]
    else:
        raise ValueError(f"Invalid rep value: {rep}. Use 'X' or 'normalized'.")
        
    results_range = {}
    results_hist = {}
    gmm_curves = {marker: {} for marker in marker_list if marker in marker_to_plot}
    num_markers = sum(1 for marker in marker_list if marker in marker_to_plot)
    rows_needed = np.ceil(num_markers / subplots_per_row).astype(int)
    fig, axes = plt.subplots(rows_needed, subplots_per_row, figsize=(20, rows_needed * plot_width), dpi=dpi, squeeze=False)
    axes = axes.flatten()

    if transparent_background: 
        fig.patch.set_alpha(0.0)  # Make the figure background transparent
        for ax in axes:
            ax.patch.set_alpha(0.0)
        
    legend_handles = []
    cmap = colormaps[colormap]
    plotted_marker_index = 0

    for marker_index, marker_name in tqdm(enumerate(marker_list), total=len(marker_list)):
        if marker_name not in marker_to_plot:
            continue
        print(f"Processing marker: {marker_name}")

        min_list = []
        max_list = []
        global_min = float('inf')
        global_max = float('-inf')
        combined_data = []

        for sample_name in sample_names:
            # Extract data for the current sample and marker
            sample_mask = adata.obs['sample_id'] == sample_name
            marker_data = data_matrix[sample_mask, marker_index].flatten()
            marker_mean_intensity_scaled = preprocess_raw(marker_data)
            combined_data.append(marker_mean_intensity_scaled)

            min_val = marker_mean_intensity_scaled.min()
            max_val = marker_mean_intensity_scaled.max()
            min_list.append(min_val)
            max_list.append(max_val)

            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)

        combined_data = np.concatenate(combined_data)
        results_range[marker_name] = {"min_list": min_list, "max_list": max_list, "global_min": global_min, "global_max": global_max}

        
        
        hist_list = []
        bin_edge_list = []
        ax = axes[plotted_marker_index]
        plotted_marker_index += 1

        for idx, sample_name in enumerate(sample_names):
            sample_mask = adata.obs['sample_id'] == sample_name
            marker_data = data_matrix[sample_mask, marker_index].flatten()
            marker_mean_intensity_scaled = preprocess_raw(marker_data)

            hist, bin_edges = np.histogram(marker_mean_intensity_scaled, bins=bin_counts, range=(global_min, global_max))
            hist_list.append(hist)
            bin_edge_list.append(bin_edges)
            
            color = cmap(idx % cmap.N)  # Get the color from the specified colormap
            line, = ax.plot(bin_edges[:-1], hist, label=f'{sample_name}', alpha=0.7, linewidth=2, color=color)
            if plot_legend:
                legend_handles.append(line)

            local_threshold, x_axis, y_axis0, y_axis1 = plot_local_gmm(marker_mean_intensity_scaled, bin_counts, global_min, global_max)
                        
            gmm_curves[marker_name][sample_name] = {
                "cut_means": local_threshold,
                "x_axis": x_axis,
                "y_axis0": y_axis0,
                "y_axis1": y_axis1
            }

            if plot_local_threshold:
                ax.axvline(local_threshold, color=color, linestyle='--', linewidth=1)


        if plot_global_threshold: 
            global_threshold = plot_global_gmm(combined_data)  
            print(f"global threshold is {global_threshold}")
            ax.axvline(global_threshold, color='black', linestyle='--', linewidth=1, label='Global Threshold')

        
        results_hist[marker_name] = {"hist_list": hist_list, "bin_edge_list": bin_edge_list}
        ax.set_title(f'{marker_name}', fontsize=16)
        ax.set_xlabel('Log10 Cell Mean Intensity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

        if xlims and marker_name in xlims:
            ax.set_xlim(xlims[marker_name])
        if ylims and marker_name in ylims:
            ax.set_ylim(ylims[marker_name])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.grid(False)
        
        if ax.legend_:
            ax.legend_.remove()

    for i in range(num_markers, len(axes)):
        axes[i].axis('off')

    if plot_legend:
        fig.legend(handles=legend_handles[:len(sample_names)], labels=sample_names, loc='center right', fontsize=14, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        print(f"Figure saved as '{save_filename}' in current working directory")
        
    plt.show()
    return results_range, results_hist, gmm_curves
