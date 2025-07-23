# -*- coding: utf-8 -*-
# src/normalization.py
# © 2025 Kunlun Wang <wangmar@ohsu.edu> — MIT License

"""
Core normalization workflows for the UniFORM pipeline.
"""
from typing import List, Dict, Optional, Tuple, Union
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import os
import pickle
from anndata import AnnData
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    "plot_line_histogram",
    "plot_correlations_and_fit_line", 
    "plot_gmm", 
    "calculate_shift_in_log_pixels", 
    "generate_normalized_feature"
]


def plot_line_histogram(
    ax: Axes,
    image: np.ndarray,
    label: str,
    alpha: float = 0.9,
    n_bins: int = 1024
) -> None:
    """
    Plot a line‐style histogram of intensity values on the provided Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to draw the histogram line.
    image : numpy.ndarray
        1D or nD array of intensity values to histogram (will be flattened).
    label : str
        Legend label for the line.
    alpha : float, optional
        Line opacity between 0 (transparent) and 1 (opaque). Default is 0.9.
    n_bins : int, optional
        Number of equally spaced bins to use in the histogram. Default is 1024.

    Returns
    -------
    None
    """
    hist, bins = np.histogram(
        image.ravel(),
        bins=n_bins,
        range=(image.min(), image.max())
    )
    ax.plot(bins[:-1], hist, label=label, alpha=alpha)


def plot_correlations_and_fit_line(
    original_intensity: np.ndarray,
    normalized_intensity: np.ndarray,
    title: str = 'Scatter Plot',
    dpi: int = 300,
    xlabel: str = 'Original Intensity',
    ylabel: str = 'Normalized Intensity'
) -> None:
    """
    Create a scatter plot comparing original vs. normalized intensities, 
    annotate with Spearman and Pearson correlations, and overlay both 
    the identity line and the least‐squares fit line.

    Parameters
    ----------
    original_intensity : np.ndarray
        1D array of raw intensity values.
    normalized_intensity : np.ndarray
        1D array of intensity values after normalization.
    title : str, optional
        Title displayed at the top of the plot. Default is 'Scatter Plot'.
    dpi : int, optional
        Dots‐per‐inch resolution of the figure. Default is 300.
    xlabel : str, optional
        Label for the x‐axis. Default is 'Original Intensity'.
    ylabel : str, optional
        Label for the y‐axis. Default is 'Normalized Intensity'.

    Returns
    -------
    None
        Displays the plot; does not return any value.
    """
    # Calculate Spearman correlation
    spearman_corr, _ = stats.spearmanr(original_intensity, normalized_intensity)

    # Calculate Pearson correlation and R^2 value
    pearson_corr, _ = stats.pearsonr(original_intensity, normalized_intensity)
    r_squared = pearson_corr**2

    # Plotting
    plt.figure(figsize=(6, 4), dpi=dpi)
    plt.scatter(original_intensity, normalized_intensity)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.grid(True)

    # Annotating Spearman correlation and R^2 values
    plt.text(0.05, 0.95, f'Spearman Correlation: {spearman_corr:.5f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'R² Value: {r_squared:.5f}', transform=plt.gca().transAxes)

    # Find the overall min and max points for the line
    pt_min = min(np.min(original_intensity), np.min(normalized_intensity))
    pt_max = max(np.max(original_intensity), np.max(normalized_intensity))

    # Plot the y=x line
    plt.plot([pt_min, pt_max], [pt_min, pt_max], 'b--')

    # Calculate the line of best fit
    slope, intercept, _, _, _ = stats.linregress(original_intensity, normalized_intensity)

    # Generate x values from the min to max observed values
    x_vals = np.linspace(np.min(original_intensity), np.max(original_intensity), 100)

    # Calculate the y values based on the slope and intercept
    y_vals = slope * x_vals + intercept

    # Plot the line of best fit
    plt.plot(x_vals, y_vals, 'r-', label=f'Best Fit Line: y={slope:.2f}x+{intercept:.2f}')

    plt.text(pt_max, pt_max, f'y={slope:.2f}x+{intercept:.2f}', color='red', verticalalignment='bottom', horizontalalignment='right')

    # Display the plot
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_gmm(
    ax: Axes,
    X: np.ndarray,
    title: str,
    marker_name: str,
    dpi: int = 300,
    xlims: Optional[Tuple[float, float]] = None
) -> float:
    """
    Fit a two‐component Gaussian Mixture Model to log‐transformed data,
    plot the histogram with overlaid GMM components and threshold,
    and return the threshold in the original scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the histogram and GMM curves.
    X : np.ndarray
        1D array of raw intensity values.
    title : str
        Title for the plot.
    marker_name : str
        Name of the marker, used in axis labeling.
    dpi : int, optional
        Figure resolution in dots per inch (not used directly on ax).
    xlims : tuple of (float, float), optional
        X-axis limits for the plot.

    Returns
    -------
    float
        The cutoff threshold in the original intensity scale, 
        corresponding to the maximum of the “negative” GMM component.
    """
    # log transform
    X = np.log(X)
    X[X == -np.inf] = 0

    gmm = GaussianMixture(
        n_components=2,
        init_params='random_from_data',
        max_iter=1000,
        random_state=10,
        covariance_type='full'
    )
    gmm.fit(X)

    preds  = gmm.predict(X)
    class0 = X[preds == 0]
    class1 = X[preds == 1]

    if class0.size == 0: class0 = np.array([0])
    if class1.size == 0: class1 = np.array([0])

    # decide which is “negative” vs “positive” by max value
    neg, pos = (class1, class0) if max(class0) > max(class1) else (class0, class1)

    # ensure cut is a scalar
    cut = float(max(neg)) 

    x_axis = np.arange(X.min(), X.max(), 0.1)
    y0 = norm.pdf(x_axis, gmm.means_[0], np.sqrt(gmm.covariances_[0])) * gmm.weights_[0]
    y1 = norm.pdf(x_axis, gmm.means_[1], np.sqrt(gmm.covariances_[1])) * gmm.weights_[1]

    # draw
    ax.hist(X, density=True, facecolor='black', alpha=0.7, bins=200)
    ax.plot(x_axis, y0[0], c='red',   label='GMM comp 0')
    ax.plot(x_axis, y1[0], c='green', label='GMM comp 1')
    # ax.axvline(cut, c='blue', label=f'cut(orig) = {np.exp(cut):.2f}, cut(log) = {cut: .2f}')
    ax.axvline(cut, c='blue', label='threshold')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f'log({marker_name})', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    if xlims: ax.set_xlim(xlims)

    return np.exp(cut)


def calculate_shift_in_log_pixels(
    intensity_ranges: Dict[str, Dict[str, float]],
    markers: List[str],
    num_bins: int,
    shifts_map: Dict[str, List[int]]
) -> Dict[str, List[float]]:
    """
    Translate FFT‐derived bin shifts into actual shifts in log‐pixel space for each marker.

    Given the global intensity range for each marker (on a log scale) and the number of
    histogram bins, this function converts each integer bin shift into an equivalent
    shift in log‐pixel units by multiplying by the bin resolution.

    Parameters
    ----------
    intensity_ranges : dict
        Mapping each marker name to a dict with keys:
          - 'global_min': float, lowest log‐intensity observed
          - 'global_max': float, highest log‐intensity observed
    markers : list of str
        List of marker names to process.
    num_bins : int
        Total number of bins used in the histogram; determines resolution.
    shifts_map : dict
        Mapping each marker name to a list of integer shifts (one per sample)
        obtained from FFT‐based cross‐correlation.

    Returns
    -------
    Dict[str, List[float]]
        For each marker, a list of shifts in log‐pixel units (floats), corresponding
        to the original integer bin shifts.
    """

    shift_in_orig_scale_dict: Dict[str, List[float]] = {}
    for marker in markers:
        # print(f"********** Processing marker {key} **********")
        min_val = intensity_ranges[marker]['global_min']
        max_val = intensity_ranges[marker]['global_max']
        increment = (max_val - min_val) / (num_bins - 1)
        shifts = shifts_map[marker]
        shift_in_log_pixels = [shift * increment for shift in shifts]
        shift_in_orig_scale_dict[marker] = shift_in_log_pixels
    return shift_in_orig_scale_dict


def generate_normalized_feature(
    feature_input: Union[List[Dict[str, np.ndarray]], AnnData],
    sample_ids: Optional[List[str]],
    markers: Optional[List[str]],
    intensity_ranges: Dict[str, Dict[str, float]],
    shifts_map: Dict[str, List[int]],
    chosen_references: List[str],
    num_bins: int,
    dpi: int = 300,
    plot_dist: bool = True,
    plot_single_cell_corr: bool = True,
    gmm_analysis: bool = True,
    save_normalized_features: bool = True
) -> None:
    """
    Normalize per-sample, per-marker intensity features according to precomputed shifts,
    then optionally visualize and save the results. Supports both:
      - Pickle-based input (list of dicts with 'intensity_mean'), and
      - AnnData-based input (writes into adata.layers['normalized']).

    Parameters
    ----------
    feature_input
        List of dicts each with 'intensity_mean': ndarray (n_markers × n_cells),
        or an AnnData whose .X is (n_cells_total × n_markers) and obs['sample_id'] labels.
    sample_ids
        List of sample identifiers for pickle mode; ignored if AnnData.
    markers
        List of marker names for pickle mode; ignored if AnnData.
    intensity_ranges
        Dict[marker] → {'min_list', 'max_list', 'global_min', 'global_max'}.
    shifts_map
        Dict[marker] → list of integer shifts (one per sample).
    chosen_references
        List of reference sample IDs per marker (length = n_markers).
    num_bins
        Number of histogram bins (for plotting).
    dpi
        Resolution for Matplotlib figures.
    plot_dist
        If True, produce distribution plots (log scale).
    plot_single_cell_corr
        If True, produce single-cell scatter plots (log scale).
    gmm_analysis
        If True, produce side-by-side GMM threshold plots.
    save_normalized_features
        If True, save per-sample results (pickles or AnnData layer).
    """
    is_adata = isinstance(feature_input, AnnData)
    if is_adata:
        adata = feature_input
        # infer sample_ids & markers
        sample_ids = (
            adata.obs['sample_id'].cat.categories.tolist()
            if hasattr(adata.obs['sample_id'], 'cat')
            else adata.obs['sample_id'].unique().tolist()
        )
        markers = adata.var['marker_name'].tolist()
        # prepare container for normalized data
        normalized_matrix = np.zeros_like(adata.X)
        # build list-of-dicts like pickle mode
        feature_data = []
        for sid in sample_ids:
            mask = adata.obs['sample_id'] == sid
            mat  = adata.X[mask, :]
            if hasattr(mat, "toarray"):
                mat = mat.toarray()
            feature_data.append({'intensity_mean': mat.T})
    else:
        feature_data = feature_input  # type: ignore
        if sample_ids is None or markers is None:
            raise ValueError("Must supply sample_ids and markers for pickle mode.")

    n_samples = len(sample_ids)
    n_markers = len(markers)

    # 1) Consistency checks
    if set(markers) != set(intensity_ranges.keys()):
        raise ValueError("Markers vs intensity_ranges keys mismatch.")
    if set(markers) != set(shifts_map.keys()):
        raise ValueError("Markers vs shifts_map keys mismatch.")
    if len(chosen_references) != n_markers:
        raise ValueError("chosen_references length must equal number of markers.")

    # 2) Per-mode checks
    if not is_adata:
        if len(feature_data) != n_samples:
            raise ValueError("feature_data length must match sample_ids length.")
        for idx, sample in enumerate(feature_data):
            if sample['intensity_mean'].shape[0] != n_markers:
                raise ValueError(
                    f"Sample '{sample_ids[idx]}' has "
                    f"{sample['intensity_mean'].shape[0]} markers; expected {n_markers}."
                )
    else:
        if adata.X.shape[1] != n_markers:
            raise ValueError("AnnData.var does not match X dimensions.")
        
    # 3) Compute pixel-scale shifts and negated scale factors
    
    for key, value in shifts_map.items():
        if isinstance(value, np.ndarray):
            shifts_map[key] = value.tolist()
            
    shift_in_pixels = calculate_shift_in_log_pixels(
        intensity_ranges=intensity_ranges,
        markers=markers,
        num_bins=num_bins,
        shifts_map=shifts_map
    )
    negated_factors = {
        marker: [-x for x in pix_shifts]
        for marker, pix_shifts in shift_in_pixels.items()
    }

    # 4) Prepare output
    if not is_adata:
        os.makedirs("normalized_data_pickle", exist_ok=True)

    print("##############################Performing feature normalization##############################\n")
    # 5) Loop over samples
    for s_idx, sid in enumerate(tqdm(sample_ids, desc="Samples", total=len(sample_ids))):
        print(f"******************************  Normalizing {sid} ******************************")
        raw_dict = feature_data[s_idx]['intensity_mean']  # shape (n_markers, n_cells)
        normalized_list: List[np.ndarray] = []

        for m_idx, marker in enumerate(markers):
            print(f"⏳ {marker} ...")
            scale = np.exp(negated_factors[marker][s_idx])
            raw_vals = raw_dict[m_idx]
            norm_vals = raw_vals * scale
            
            print(f"⏳ Normalizing {marker} | reference={chosen_references[m_idx]} | scale factor={scale}")

            # visualization (same logic as before; assumes helper funcs exist)
            if plot_dist:
                raw_log  = log_transform_intensities(raw_vals)
                norm_log = log_transform_intensities(norm_vals)
                ref_sid  = chosen_references[m_idx]
                ref_idx  = sample_ids.index(ref_sid)
                ref_raw  = feature_data[ref_idx]['intensity_mean'][m_idx]
                ref_log  = log_transform_intensities(ref_raw)

                fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=dpi)
                plot_line_histogram(ax, raw_log,  f"{sid} {marker} – Original (log)", n_bins=num_bins, alpha=0.5)
                plot_line_histogram(ax, norm_log, f"{sid} {marker} – Normalized (log)", n_bins=num_bins)
                plot_line_histogram(ax, ref_log,  f"{ref_sid} {marker} – Reference (log)", n_bins=num_bins, alpha=0.5)
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
                ax.set_title(f"{sid} {marker} Mean Intensity (log)")
                ax.set_xlabel("Log(Cell Mean Intensity)")
                ax.set_ylabel("Frequency")
                plt.show()

            if plot_single_cell_corr:
                raw_log  = log_transform_intensities(raw_vals, min_value=0)
                norm_log = log_transform_intensities(norm_vals, min_value=0)
                plot_correlations_and_fit_line(
                    raw_log, norm_log,
                    title=f"{sid} {marker} Original vs Normalized (log)",
                    dpi=dpi,
                    xlabel="Original (log)",
                    ylabel="Normalized (log)"
                )

            if gmm_analysis:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
                fig.subplots_adjust(wspace=0.3)
                thr_o = plot_gmm(ax1, raw_vals.reshape(-1,1), title=f"{sid} {marker} – Original", marker_name=marker, dpi=dpi)
                thr_n = plot_gmm(ax2, norm_vals.reshape(-1,1), title=f"{sid} {marker} – Normalized", marker_name=marker, dpi=dpi)
                handles, labels = ax2.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
                plt.show()
                cond_o, cond_n = raw_vals>=thr_o, norm_vals>=thr_n
                print(f"\nGMM for {sid}–{marker}\n"
                      f"  Orig: thr={thr_o:.2f} ({np.log(thr_o):.2f} in log), pos={cond_o.sum()}/{len(raw_vals)} ({cond_o.mean()*100:.1f}%)\n"
                      f"  Norm: thr={thr_n:.2f} ({np.log(thr_n):.2f} in log), pos={cond_n.sum()}/{len(norm_vals)} ({cond_n.mean()*100:.1f}%)\n")

            normalized_list.append(norm_vals)

        # save or assign
        if save_normalized_features:
            if is_adata:
                mask = adata.obs['sample_id'] == sid
                arr  = np.stack(normalized_list, axis=0).T  # (cells, markers)
                normalized_matrix[mask, :] = arr
        
            else:
                stacked = np.stack(normalized_list, axis=0)
                path    = os.path.join("normalized_data_pickle", f"{sid}_normalized_feature.pkl")
                with open(path, 'wb') as f:
                    pickle.dump({'intensity_mean': stacked}, f)
                print(f"✅ Saved {sid} normalized features to {path}")

    if save_normalized_features and is_adata:
        adata.layers['normalized'] = normalized_matrix
        print("✅ Saved normalized data to adata.layers['normalized'].")

        output_fname = "Anndata_Normalized.h5ad"
        adata.write(output_fname)
        print(f"✅ Wrote AnnData to '{output_fname}'.")

    print("🎉 Feature normalization complete!")