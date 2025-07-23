# -*- coding: utf-8 -*-
# src/registration.py
# © 2025 Kunlun Wang <wangmar@ohsu.edu> — MIT License

"""
Registration routines for UniFORM normalization pipeline.

Provides functions to compute per‐sample shifts
"""

import numpy as np
import matplotlib.pyplot as plt
import skfda
from scipy.signal import correlate
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    "correct_shifted_histograms",
    "plot_histogram_distributions",
    "compute_landmark_shifts", 
    "compute_correlation_shifts", 
    "automatic_registration"
]


def correct_shifted_histograms(
    original_fd: skfda.FDataGrid,
    shifted_fd: skfda.FDataGrid,
    shift_values: List[int]
) -> None:
    """
    Restore edge values after histogram shifts to prevent artifacts.

    Parameters
    ----------
    original_fd : FDataGrid
        The original unshifted histogram data.
    shifted_fd : FDataGrid
        The shifted histogram data to correct.
    shift_values : List[int]
        Integer shift applied to each sample histogram.
    """
    for sample_idx, shift in enumerate(shift_values):
        # Always copy the first bin from the original
        shifted_fd.data_matrix[sample_idx][0] = original_fd.data_matrix[sample_idx][0]
        if shift < 0:
            # Zero-out the tail bin when shifting left
            tail_index = min(-shift, shifted_fd.data_matrix[sample_idx].shape[0] - 1)
            shifted_fd.data_matrix[sample_idx][tail_index] = np.array([0.])


def plot_histogram_distributions(
    fd: skfda.FDataGrid,
    sample_ids: List[str],
    grid_points: np.ndarray,
    marker_name: str,
    title_suffix: str,
    xlim_index: int,
    x_limits: Optional[List[Tuple[float, float]]],
    fig: plt.Figure,
    ax: plt.Axes,
    reference_id: Optional[str],
    color_map: Dict[str, Tuple[float, float, float]]
) -> None:
    """
    Plot multiple sample histograms from an FDataGrid, highlighting a reference.

    Parameters
    ----------
    fd : FDataGrid
        Histogram data (n_samples × n_bins × 1).
    sample_ids : List[str]
        Ordered sample identifiers.
    grid_points : np.ndarray
        X-axis values for the histograms.
    marker_name : str
        Name of the marker for title.
    title_suffix : str
        Suffix to append to the plot title.
    xlim_index : int
        Index into x_limits for this plot.
    x_limits : List[(min, max)] or None
        Optional x-axis limits per marker.
    fig : matplotlib.figure.Figure
        Figure object (unused but kept for compatibility).
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    reference_id : str or None
        Sample to highlight in black.
    color_map : Dict[str, tuple]
        Mapping sample_id → RGB color tuple.
    """
    ax.set_title(f"{marker_name} {title_suffix}", fontweight='bold')
    for idx, sid in enumerate(sample_ids):
        line_color = 'black' if sid == reference_id else color_map[sid]
        line_width = 2.5 if sid == reference_id else 1.5
        ax.plot(
            grid_points,
            fd.data_matrix[idx, :, 0],
            label=sid,
            color=line_color,
            linewidth=line_width,
            alpha=0.7
        )
    ax.set_xlabel('Grid Points', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pixel Counts', fontsize=12, fontweight='bold')
    if x_limits and xlim_index < len(x_limits):
        ax.set_xlim(x_limits[xlim_index])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.tick_params(width=2)
    ax.legend().remove()


def compute_landmark_shifts(
    fd: skfda.FDataGrid,
    landmark_positions: List[float],
    anchor_position: Optional[float]
) -> List[float]:
    """
    Compute per-sample shifts based on landmark deviations.

    Parameters
    ----------
    fd : FDataGrid
        Histogram data (n_samples × n_bins × 1).
    landmark_positions : List[float]
        User-chosen landmark for each sample.
    anchor_position : float or None
        If None, use mean of landmark_positions; otherwise align to this.

    Returns
    -------
    List[float]
        Shift values for each sample.
    """
    landmarks = np.atleast_1d(landmark_positions)
    if landmarks.size != fd.n_samples:
        raise ValueError(
            f"Landmark list length ({landmarks.size}) must match number of samples ({fd.n_samples})."
        )
    ref = np.mean(landmarks) if anchor_position is None else anchor_position
    return (landmarks - ref).tolist()


def compute_correlation_shifts(
    reference_hist: np.ndarray,
    histogram_list: List[np.ndarray]
) -> Tuple[List[int], List[int]]:
    """
    Compute direct and FFT-based shift estimates by cross-correlation.

    Parameters
    ----------
    reference_hist : np.ndarray
        1D reference histogram.
    histogram_list : List[np.ndarray]
        Histograms to align against the reference.

    Returns
    -------
    direct_shifts : List[int]
        Shifts from direct method.
    fft_shifts    : List[int]
        Shifts from FFT-based method.
    """
    direct_shifts: List[int] = []
    fft_shifts: List[int] = []
    for hist in histogram_list:
        corr_direct = correlate(hist.flatten(), reference_hist.flatten(), mode='full', method='direct')
        corr_fft    = correlate(hist.flatten(), reference_hist.flatten(), mode='full', method='fft')
        shift_direct = np.argmax(corr_direct) - (len(reference_hist) - 1)
        shift_fft    = np.argmax(corr_fft)    - (len(reference_hist) - 1)
        direct_shifts.append(int(shift_direct))
        fft_shifts.append(int(shift_fft))
    return direct_shifts, fft_shifts


def automatic_registration(
    histogram_data: Dict[str, Dict[str, List[np.ndarray]]],
    all_markers: List[str],
    selected_markers: List[str],
    sample_ids: List[str],
    reference_samples: Optional[List[str]],
    landmark_map: Optional[Dict[str, List[float]]],
    num_bins: int,
    dpi: int = 300,
    x_limits: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Dict[str, List[int]], List[str]]:
    """
    Align per-marker histograms across samples, visualize both original and
    shifted distributions, and compute inferred landmark positions.

    For each marker in `selected_markers`, this function will:
      1. Determine a reference sample (either user‑provided or chosen by
         proximity in landmark space or histogram space).
      2. Build a functional data grid of all sample histograms.
      3. Plot the original distributions side‑by‑side with the normalized
         (shifted) distributions.
      4. Compute integer shifts (in bins) using either landmark registration
         or correlation-based alignment.
      5. Auto‑compute the reference landmark as the peak bin of the reference
         histogram.
      6. Derive implied landmark positions for all samples given those shifts.
      7. Return the computed shift map, the list of chosen references, and the
         implied landmark map for downstream refinement.

    Parameters
    ----------
    histogram_data : dict
        Mapping marker → {'counts': [array(hist_counts) per sample], 
        'bin_edges': [...]}.  Used to construct FDataGrid objects.
    all_markers : list of str
        Full ordered list of markers in the pipeline.
    selected_markers : list of str
        Subset of `all_markers` to process and plot.
    sample_ids : list of str
        Ordered list of sample identifiers corresponding to rows in each histogram.
    reference_samples : list of str or None
        If provided, reference_samples[i] is the user‑supplied reference for
        selected_markers[i]; otherwise None triggers automatic selection.
    landmark_map : dict or None
        Mapping marker → list of numeric landmark positions.  If provided,
        landmark registration is used; otherwise alignment falls back to
        cross‑correlation of histograms.
    num_bins : int
        Number of bins in each histogram (defines the functional grid length).
    dpi : int, optional
        Resolution for any Matplotlib figures.  Default is 300.
    x_limits : list of (min,max) tuples, optional
        If provided, x_limits[i] sets the x‑axis bounds for selected_markers[i].

    Returns
    -------
    shift_map : dict
        Mapping each marker to its list of integer bin shifts for all samples.
    chosen_refs : list of str
        The actual reference sample IDs selected or used for each marker.
    implied_landmarks_map : dict
        Mapping each marker to the implied integer landmark locations for all
        samples, computed from the reference landmark and shifts.
    """
    t = np.linspace(0, num_bins - 1, num_bins)
    shift_map: Dict[str, List[int]] = {}
    chosen_refs: List[str] = []
    implied_landmarks_map: Dict[str, List[int]] = {}

    cmap = plt.get_cmap('tab20')
    color_map = {sid: cmap(i % 20) for i, sid in enumerate(sample_ids)}

    for idx, marker in enumerate(selected_markers):
        print(f"\n-------------------- Processing marker \033[1m'{marker}'\033[0m --------------------")

        # determine reference
        if reference_samples is not None:
            ref_id = reference_samples[all_markers.index(marker)]
            ref_idx = sample_ids.index(ref_id)
            print(f"Reference sample → '{ref_id}' (idx {ref_idx})")
        else:
            ref_id = None
            ref_idx = None
            print("☑️ \033[1mNo reference provided\033[0m → will select by \033[1mproximity\033[0m")

        # build FD grid
        fd = skfda.FDataGrid(
            data_matrix=histogram_data[marker]['counts'],
            sample_points=t,
            extrapolation="zeros"
        )

        # plot original
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi)
        fig.subplots_adjust(wspace=0.4)
        plot_histogram_distributions(
            fd, sample_ids, t, marker, "Original Distributions",
            idx, x_limits, fig, ax1, ref_id, color_map
        )

        # compute shift
        lm_positions = landmark_map.get(marker) if landmark_map is not None else None
        if lm_positions is not None:
            if reference_samples is None:
                mean_pos = np.mean(lm_positions)
                distances = np.abs(np.array(lm_positions) - mean_pos)
                ref_idx = int(np.argmin(distances))
                ref_id = sample_ids[ref_idx]
                print(f"✅ \033[1mLandmarks provided\033[0m → selected \033[1m'{ref_id}'\033[0m closest to mean {mean_pos:.2f}")
                raw_shifts = compute_landmark_shifts(fd, lm_positions, lm_positions[ref_idx])
            else:
                print(f"✅ \033[1mLandmarks provided\033[0m → aligning to \033[1m'{ref_id}'\033[0m landmark")
                raw_shifts = compute_landmark_shifts(fd, lm_positions, lm_positions[ref_idx])
        else:
            if reference_samples is None:
                data_matrix = fd.data_matrix[:, :, 0]
                mean_hist = np.mean(data_matrix, axis=0)
                distances = np.linalg.norm(data_matrix - mean_hist, axis=1)
                ref_idx = int(np.argmin(distances))
                ref_id = sample_ids[ref_idx]
                print(f"☑️ \033[1mNo landmarks\033[0m → automatic alignment to mean, selecting \033[1m'{ref_id}'\033[0m by histogram proximity \n")
                ref_hist = data_matrix[ref_idx]
                _, raw_shifts = compute_correlation_shifts(ref_hist, data_matrix)
            else:
                ref_hist = fd.data_matrix[ref_idx]
                _, raw_shifts = compute_correlation_shifts(ref_hist, fd.data_matrix)
                print(f"☑️ \033[1mNo landmarks\033[0m → correlating to \033[1m'{ref_id}'\033[0m histogram \n")

        chosen_refs.append(ref_id)
        shifts = np.round(raw_shifts).astype(int).tolist()
        print(f"✨ \033[1mComputed shifts\033[0m: {shifts}")

        # === INSERTED: Compute reference landmark from FDataGrid histogram ===
        # Use the FDGrid counts and sample_points (t)
        ref_counts   = fd.data_matrix[ref_idx, :, 0]
        grid_points  = fd.sample_points[0]
        skip_bins    = 1   # skip low-index bins to avoid zeros
        search_counts = ref_counts[skip_bins:]
        peak_rel     = int(np.argmax(search_counts))
        peak_idx     = peak_rel + skip_bins
        # landmark = grid point at the peak
        ref_landmark = int(round(grid_points[peak_idx]))
        print(f"✨ \033[1mAuto-computed reference landmark\033[0m for \033[1m'{marker}'\033[0m on \033[1m'{ref_id}'\033[0m: {ref_landmark}")

        # === INSERTED: Compute implied landmark locations for all samples ===
        bin_width = grid_points[1] - grid_points[0]
    
        implied_landmarks = [
            int(round(ref_landmark + s * bin_width))
            for s in shifts
        ]

        print(f"✨ \033[1mImplied landmark positions\033[0m for \033[1m'{marker}'\033[0m: {implied_landmarks}")

        implied_landmarks_map[marker] = implied_landmarks
        shift_map[marker] = shifts

        # apply shifts and correct
        fd_shifted = fd.shift(shifts, restrict_domain=False)
        correct_shifted_histograms(fd, fd_shifted, shifts)

        # plot normalized
        plot_histogram_distributions(
            fd_shifted, sample_ids, t, marker, "Normalized Distributions",
            idx, x_limits, fig, ax2, ref_id, color_map
        )

        # legend
        handles, labels = ax1.get_legend_handles_labels()
        if ref_idx is not None:
            ref_line = plt.Line2D([], [], color='black', lw=2, label=f"{ref_id} (reference)")
            others = [(h,l) for h,l in zip(handles, labels) if l != ref_id]
            oh, ol = zip(*others)
            fig.legend([ref_line, *oh], [ref_line.get_label(), *ol], loc='center left', bbox_to_anchor=(1,0.5))
        else:
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))

        plt.show()

    return shift_map, chosen_refs, implied_landmarks_map