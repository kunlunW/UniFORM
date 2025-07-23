# -*- coding: utf-8 -*-
# src/preprocessing.py
# © 2025 Kunlun Wang <wangmar@ohsu.edu> — MIT License

"""
Preprocessing utilities for the UniFORM normalization pipeline.

This module contains routines to prepare raw intensity values into histogram distribution
"""

from typing import List, Dict, Optional, Tuple, Union
import itertools
import numpy as np
import matplotlib.pyplot as plt
import anndata
from scipy import sparse
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    "log_transform_intensities",
    "fit_two_component_gmm",
    "process_sample_distributions"
]


def log_transform_intensities(
    intensities: np.ndarray,
    min_value: float = 1.0
) -> np.ndarray:
    """
    Apply a log transformation to a 1D array of intensity values.

    This function:
      1. Filters out any values below `min_value` (to avoid log of zero/negative).
      2. Applies the natural logarithm to the remaining values.
      3. Replaces any resulting -inf entries with 0.

    Parameters
    ----------
    intensities : np.ndarray
        Array of raw intensity values.
    min_value : float, optional (default=1.0)
        Minimum threshold for values to be retained before log-transform.

    Returns
    -------
    np.ndarray
        Log-transformed intensity values, with -inf replaced by 0.
    """
    # 1) Filter out values below threshold
    valid_mask = intensities >= min_value
    filtered = intensities[valid_mask]

    # 2) Take natural log
    log_values = np.log(filtered)

    # 3) Replace any -inf (if any) with 0
    log_values[np.isneginf(log_values)] = 0.0

    return log_values

    
def fit_two_component_gmm(
    intensities: np.ndarray,
    num_bins: int,
    min_value: float,
    max_value: float,
    verbose: bool = False
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a two-component Gaussian Mixture Model (GMM) to intensity data and compute a cutoff threshold.

    Parameters
    ----------
    intensities : np.ndarray
        1D array of intensity values (e.g., log-transformed), shape (n_samples,).
    num_bins : int
        Number of evenly spaced points on which to evaluate the mixture model PDFs.
    min_value : float
        Minimum value of the evaluation grid (global minimum intensity).
    max_value : float
        Maximum value of the evaluation grid (global maximum intensity).
    verbose : bool, optional
        If True, print per-sample and GMM warning messages;
        if False, suppress prints and show tqdm bars.
        
    Returns
    -------
    threshold : float
        The cutoff value that best separates the two Gaussian components.
    eval_points : np.ndarray
        1D array of length `num_bins` of evaluation grid points between `min_value` and `max_value`.
    pdf_comp0 : np.ndarray
        PDF values of GMM component 0 at `eval_points`.
    pdf_comp1 : np.ndarray
        PDF values of GMM component 1 at `eval_points`.
    """
    # Ensure correct shape for GMM fitting
    data_reshaped = intensities.reshape(-1, 1)

    # Fit the GMM
    gmm_model = GaussianMixture(
        n_components=2,
        init_params='random_from_data',
        max_iter=1000,
        random_state=10,
        covariance_type='full'
    )
    gmm_model.fit(data_reshaped)

    # Assign cluster labels
    labels = gmm_model.predict(data_reshaped)
    cluster_0 = data_reshaped[labels == 0]
    cluster_1 = data_reshaped[labels == 1]

    # Handle possible empty clusters
    if cluster_0.size == 0:
        if verbose:
            print("Warning: Cluster 0 is empty; using fallback value 0.")
        cluster_0 = np.array([0])
    if cluster_1.size == 0:
        if verbose:
            print("Warning: Cluster 1 is empty; using fallback value 0.")
        cluster_1 = np.array([0])

    # Determine negative cluster (lower maximum)
    neg_cluster, pos_cluster = (
        (cluster_1, cluster_0) if cluster_0.max() < cluster_1.max()
        else (cluster_0, cluster_1)
    )

    # Create evaluation grid
    eval_points = np.linspace(min_value, max_value, num_bins)

    # Extract GMM parameters and compute PDFs
    mean0, cov0, w0 = (
        gmm_model.means_[0].item(),
        gmm_model.covariances_[0].item(),
        gmm_model.weights_[0].item()
    )
    mean1, cov1, w1 = (
        gmm_model.means_[1].item(),
        gmm_model.covariances_[1].item(),
        gmm_model.weights_[1].item()
    )
    pdf_comp0 = norm.pdf(eval_points, mean0, np.sqrt(cov0)) * w0
    pdf_comp1 = norm.pdf(eval_points, mean1, np.sqrt(cov1)) * w1

    # Threshold is the max of the negative cluster
    threshold = float(np.max(neg_cluster))

    return threshold, eval_points, pdf_comp0, pdf_comp1


def process_sample_distributions(
    feature_input: Union[List[Dict[str, np.ndarray]], anndata.AnnData],
    sample_ids: Optional[List[str]]       = None,
    all_markers: Optional[List[str]]      = None,
    markers_to_plot: Optional[List[str]]  = None,
    use_normalized: bool                  = False, 
    num_bins: int                         = 1024,
    plots_per_row: int                    = 4,
    dpi: int                              = 300,
    xlims: Optional[List[Tuple[float, float]]] = None,
    ylims: Optional[List[Tuple[float, float]]] = None,
    output_figure_path: Optional[str]     = None,
    verbose: bool                         = False
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]]
]:
    """
    Compute per-sample log‐intensity ranges, histograms, and 2-component GMM fits.

    This function accepts either:
      1) A list of dicts, each with key 'intensity_mean' mapping to an ndarray
         of shape (n_markers, n_cells) per sample, or
      2) An AnnData object with raw intensities in `.X` (n_obs × n_vars), and
         a categorical `sample_id` column in `.obs`, and marker names in
         `.var['marker_name']`.

    It performs the following steps for each marker:
      - Log‐transforms each sample’s cell‐mean intensities.
      - Computes per-sample min/max and a global range.
      - Plots all sample histograms on a common axis.
      - Fits a two‐component Gaussian Mixture Model to each sample’s distribution.
      - Returns:
        • intensity_ranges[marker] = {
            'min_list': [...],
            'max_list': [...],
            'global_min': float,
            'global_max': float
          }
        • histograms[marker] = {
            'bin_edges': [array(...) per sample],
            'counts':    [array(...) per sample]
          }
        • gmm_models[marker][sample_id] = {
            'cut':    float (exp of log‐cut),
            'x_axis': ndarray,
            'y0':     ndarray,
            'y1':     ndarray
          }

    Parameters
    ----------
    feature_input
        Either a list of per-sample dicts ({'intensity_mean': ndarray}), or
        an AnnData object. If AnnData, `.X` must contain raw intensities,
        `.obs['sample_id']` the sample labels, and `.var['marker_name']`
        the marker names.
    sample_ids
        List of sample identifiers; ignored when `feature_input` is AnnData.
    all_markers
        Ordered list of all marker names. Required if `feature_input` is a list.
    markers_to_plot
        Subset of `all_markers` to process and plot; defaults to `all_markers`.
    use_normalized
        if the input datatype is anndata, whether to use the normalized layer as input 
    num_bins
        Number of bins for histogram computation.
    plots_per_row
        Number of subplot columns per row.
    dpi
        Figure resolution for plotting.
    xlims, ylims
        Optional per-marker axis limit lists, each element a (min, max) tuple.
    output_figure_path
        If provided, save the summary figure to this path.
    verbose : bool, optional
        If True, print per-sample and GMM warning messages;
        if False, suppress prints and show tqdm bars.

    Returns
    -------
    intensity_ranges : dict
        Marker→{min_list, max_list, global_min, global_max}.
    histograms : dict
        Marker→{bin_edges, counts} where each is a list across samples.
    gmm_models : dict
        Marker→Sample→GMM fit outputs (cut, x_axis, y0, y1).
    """
    # --- 1) If AnnData, infer sample_ids & all_markers, split into feature_data --- #
    if isinstance(feature_input, anndata.AnnData):
        adata = feature_input

        # derive sample IDs
        sample_ids = (
            adata.obs['sample_id'].cat.categories.tolist()
            if hasattr(adata.obs['sample_id'], 'cat')
            else adata.obs['sample_id'].unique().tolist()
        )

        # derive marker list
        all_markers = adata.var['marker_name'].tolist()

        # build feature_data list-of-dicts
        feature_data = []
        for samp in sample_ids:
            mask = adata.obs['sample_id'] == samp
            data_matrix = adata.layers["normalized"] if use_normalized else adata.X
            mat = data_matrix[mask, :]
            if sparse.issparse(mat):
                mat = mat.toarray()
            feature_data.append({'intensity_mean': mat.T})
    else:
        # user-supplied list-of-dicts
        feature_data = feature_input
        if all_markers is None:
            raise ValueError("Must supply all_markers when feature_input is a list.")

    # --- 2) Default markers_to_plot to all_markers if not provided --- #
    if markers_to_plot is None:
        markers_to_plot = all_markers

    # Initialize outputs
    intensity_ranges: Dict[str, Dict[str, List[float]]] = {}
    histograms: Dict[str, Dict[str, np.ndarray]]       = {}
    gmm_models: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        m: {} for m in markers_to_plot
    }

    # Prepare figure grid
    n_markers = len(markers_to_plot)
    n_rows    = int(np.ceil(n_markers / plots_per_row))
    fig, axes = plt.subplots(
        n_rows, plots_per_row,
        figsize=(20, n_rows * 4),
        dpi=dpi,
        squeeze=False
    )
    axes = axes.flatten()
    legend_handles = []
    plot_idx = 0

    # Color cycle
    cmap    = plt.get_cmap('tab20')
    palette = [cmap(i) for i in range(20)]
    colors  = list(itertools.islice(itertools.cycle(palette), len(sample_ids)))

    # Main loop over markers
    # markers_iter = (all_markers if verbose else tqdm(all_markers, desc="Markers"))
    markers_iter = tqdm(all_markers, desc="Markers")
    for m_idx, marker in enumerate(markers_iter):
    
        if marker not in markers_to_plot:
            continue

        # 1) Compute log-scale ranges
        min_list, max_list = [], []
        global_min, global_max = np.inf, -np.inf

        for feat, ids in zip(feature_data, sample_ids):
            if verbose:
                print(f"✅ Processing {ids} for marker {marker}......")
            arr = feat['intensity_mean'][m_idx]
            arr_log = log_transform_intensities(arr)
            mn, mx = arr_log.min(), arr_log.max()
            min_list.append(mn)
            max_list.append(mx)
            global_min = min(global_min, mn)
            global_max = max(global_max, mx)

        intensity_ranges[marker] = {
            'min_list':    min_list,
            'max_list':    max_list,
            'global_min':  global_min,
            'global_max':  global_max
        }

        # 2) Plot histograms + GMM
        ax = axes[plot_idx]
        plot_idx += 1
        ax.set_title(marker, fontsize=12, fontweight='bold')
        ax.set_xlabel('Log(Cell Mean Intensity)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(width=2)

        bin_edges_list, counts_list = [], []

        for feat, sid, col in zip(feature_data, sample_ids, colors):
            arr     = feat['intensity_mean'][m_idx]
            arr_log = log_transform_intensities(arr)

            counts, edges = np.histogram(
                arr_log, bins=num_bins, range=(global_min, global_max)
            )
            bin_edges_list.append(edges)
            counts_list.append(counts)

            line, = ax.plot(
                edges[:-1], counts,
                color=col, alpha=0.7, linewidth=2,
                label=sid
            )
            legend_handles.append(line)

            threshold, eval_points, pdf_comp0, pdf_comp1 = fit_two_component_gmm(
                arr_log, num_bins, global_min, global_max, verbose=verbose
            )
            
            gmm_models[marker][sid] = {
                'cut':    threshold,
                'x_axis': eval_points,
                'y0':     pdf_comp0,
                'y1':     pdf_comp1
            }

        histograms[marker] = {
            'bin_edges': bin_edges_list,
            'counts':    counts_list
        }

        # Axis limits
        if xlims and m_idx < len(xlims):
            ax.set_xlim(xlims[m_idx])
        if ylims and m_idx < len(ylims):
            ax.set_ylim(ylims[m_idx])
        if ax.legend_:
            ax.legend_.remove()

    # Hide extras
    for ax in axes[plot_idx:]:
        ax.axis('off')

    # Master legend on right
    fig.legend(
        handles=legend_handles[:len(sample_ids)],
        labels=sample_ids,
        loc='center right',
        fontsize=14,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print(f"✅ Saved summary figure to: {output_figure_path}")

    plt.show()
    return intensity_ranges, histograms, gmm_models