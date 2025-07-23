# -*- coding: utf-8 -*-
# src/landmark.py
# © 2025 Kunlun Wang <wangmar@ohsu.edu> — MIT License

"""
Plotting function the optional landmark finetuning.

Contains functions to plot interactive Plotly distributions
"""

from typing import List, Optional, Dict, Tuple, Union
import skfda
import plotly.graph_objects as go
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.io as pio
from anndata import AnnData
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    "plot_distributions_plotly",
    "landmark_refinement"
]


def plot_distributions_plotly(
    fd: skfda.FDataGrid,
    sample_ids: List[str],
    grid: np.ndarray,
    marker: str,
    title_suffix: str,
    subplot_index: int,
    xlim: Optional[List[Tuple[float, float]]],
    fig: go.Figure,
    row: int,
    col: int,
    color_map: Dict[str, str],
    gmm_models: Dict[str, Dict[str, np.ndarray]]
) -> go.Figure:
    """
    Render histogram traces and overlaid GMM component curves for a set of samples 
    on a Plotly subplot.

    Parameters
    ----------
    fd
        Functional data grid containing histogram counts; shape (n_samples, n_bins, 1).
    sample_ids
        Ordered list of sample identifiers corresponding to the rows of `fd.data_matrix`.
    grid
        1D array of bin center positions matching the second dimension of `fd.data_matrix`.
    marker
        Name of the marker being plotted; used in the subplot title.
    title_suffix
        Suffix for the subplot title (e.g., "Original Distributions").
    subplot_index
        Index of the current marker in the overall sequence; used to select x-axis limits.
    xlim
        Optional list of (min, max) tuples for per-marker x-axis ranges.
    fig
        Plotly Figure object to which traces will be added.
    row, col
        Row and column indices within the figure’s subplot grid.
    color_map
        Mapping from sample_id to a hex color string.
    gmm_models
        Nested mapping: gmm_models[marker][sample_id] → dict with keys
        "x_axis", "y0", and "y1" for the two component PDFs.

    Returns
    -------
    go.Figure
        The updated Plotly Figure with added histogram and GMM traces.
    """
    title = f'{marker} {title_suffix}'
    fig.update_layout(
        title=title,
        xaxis_title='Grid Points',
        yaxis_title='Pixel Counts',
        showlegend=True,
        legend=dict(
            x=1,
            y=1,
            traceorder='normal',
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
        plot_bgcolor='white'
    )

    for index, sample_name in enumerate(sample_ids):
        color = mcolors.to_hex(color_map[sample_name])
        line_width = 1.5
        
        # Plot histogram line
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=fd.data_matrix[index, :, 0],
                mode='lines',
                line=dict(color=color, width=line_width),
                name=sample_name,
                text=[f'{sample_name}: ({x}, {y})' for x, y in zip(grid, fd.data_matrix[index, :, 0])]
            ),
            row=row, col=col
        )

        # Convert GMM curves to functional data on the same grid
        gmm_data = gmm_models[marker][sample_name]
        x_axis_fd = skfda.FDataGrid(data_matrix=gmm_data["x_axis"], sample_points=grid, extrapolation="zeros")

        # Find histogram peak value for scaling
        hist_max = np.max(fd.data_matrix[index, :, 0])

        # Find peak values in GMM curves
        gmm_max0 = np.max(gmm_data["y0"])
        gmm_max1 = np.max(gmm_data["y1"])

        # Calculate scaling factors
        scale_factor0 = hist_max / gmm_max0 if gmm_max0 > 0 else 1
        scale_factor1 = hist_max / gmm_max1 if gmm_max1 > 0 else 1

        # Scale GMM curves by peak normalization factor
        y_axis0_rescaled = gmm_data["y0"] * scale_factor0
        y_axis1_rescaled = gmm_data["y1"] * scale_factor1
        y_axis0_fd = skfda.FDataGrid(data_matrix=y_axis0_rescaled.reshape(-1), sample_points=grid, extrapolation="zeros")
        y_axis1_fd = skfda.FDataGrid(data_matrix=y_axis1_rescaled.reshape(-1), sample_points=grid, extrapolation="zeros")

        # Plot the GMM curves as dashed lines with transparency
        fig.add_trace(
            go.Scatter(
                x=x_axis_fd.sample_points[0],
                y=y_axis0_fd.data_matrix[0, :, 0],
                mode='lines',
                line=dict(color=color, width=line_width, dash='dash'),
                opacity=0.9,  # Transparency for GMM curve
                showlegend=False
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis_fd.sample_points[0],
                y=y_axis1_fd.data_matrix[0, :, 0],
                mode='lines',
                line=dict(color=color, width=line_width, dash='dash'),
                opacity=0.9,  # Transparency for GMM curve
                showlegend=False
            ),
            row=row, col=col
        )

    if xlim and subplot_index < len(xlim):
        fig.update_xaxes(range=xlim[subplot_index])

    return fig


def landmark_refinement(
    histogram_data: Dict[str, Dict[str, List[np.ndarray]]],
    gmm_models: Dict[str, Dict[str, np.ndarray]],
    markers: Optional[List[str]]        = None,
    sample_ids: Optional[List[str]]     = None,
    data_source: Optional[AnnData]      = None,
    num_bins: int                       = 1024,
    dpi: int                            = 300,
    x_limits: Optional[List[Tuple[float, float]]] = None,
    group_size: int                     = 4,
    implied_landmarks_map: Optional[Dict[str, List[int]]] = None,
    verbose: bool                       = False,
    output_directory: str               = "landmark_picking"
) -> None:
    """
    Generate HTML files for interactive landmark selection and scaffold a CSV.

    Works with either:
      • A pickle-based pipeline (provide histogram_data, gmm_models, markers & sample_ids), or
      • An AnnData-based pipeline (pass combined AnnData as data_source).

    Parameters
    ----------
    histogram_data : dict
        Mapping marker → {'bin_edges': [...], 'counts': [...]}
    gmm_models : dict
        Mapping marker → sample_id → {'x_axis','pdf0','pdf1'}
    markers : list of str, optional
        Marker names; if None and data_source is AnnData, inferred from var['marker_name']
    sample_ids : list of str, optional
        Sample identifiers; if None and data_source is AnnData, inferred from obs['sample_id']
    data_source : AnnData, optional
        AnnData to infer markers & sample_ids
    num_bins : int
        Number of grid points / histogram bins
    dpi : int
        Resolution for Matplotlib colormap
    x_limits : list of (min,max), optional
        Per-marker x-axis limits
    group_size : int
        Samples per HTML file
    implied_landmarks_map : dict, optional
        Mapping marker → implied landmark ints per sample
    verbose : bool
        If True, print progress and show tqdm; if False, only show tqdm
    output_directory : str
        Directory to write HTML & CSV

    Returns
    -------
    None
    """
    # infer markers & samples from AnnData
    if isinstance(data_source, AnnData):
        if markers is None:
            markers = data_source.var["marker_name"].tolist()
        if sample_ids is None:
            obs = data_source.obs["sample_id"]
            sample_ids = obs.cat.categories.tolist() if hasattr(obs, "cat") else obs.unique().tolist()

    if markers is None or sample_ids is None:
        raise ValueError("Must supply 'markers' and 'sample_ids' or pass AnnData via data_source.")

    os.makedirs(output_directory, exist_ok=True)

    grid = np.linspace(0, num_bins - 1, num_bins)
    cmap = plt.get_cmap("tab10")
    color_map = {sid: mcolors.to_hex(cmap(i % 10)) for i, sid in enumerate(sample_ids)}

    # iterate over groups
    group_indices = range(0, len(sample_ids), group_size)
    group_iter = tqdm(group_indices, desc="HTML files")  # track number of HTML files generated

    for start in group_iter:
        group = sample_ids[start : start + group_size]
        if not group:
            continue
        if verbose:
            print(f"Processing samples {group[0]} to {group[-1]}")

        html_body = ""
        marker_iter = enumerate(markers)  # no tqdm here, only track HTML files

        for idx, marker in marker_iter:
            if verbose:
                print(f"  Marker: {marker}")

            fig = psub.make_subplots(rows=1, cols=1, subplot_titles=[f"{marker} Distributions"], horizontal_spacing=0.1)

            counts_list = [histogram_data[marker]["counts"][sample_ids.index(sid)] for sid in group]
            fd = skfda.FDataGrid(data_matrix=np.stack(counts_list, axis=0)[:, :, None], sample_points=grid, extrapolation="zeros")

            fig = plot_distributions_plotly(
                fd=fd,
                sample_ids=group,
                grid=grid,
                marker=marker,
                title_suffix="Original Distributions",
                subplot_index=idx,
                xlim=x_limits,
                fig=fig,
                row=1,
                col=1,
                color_map=color_map,
                gmm_models=gmm_models,
            )

            html_body += pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        fname = f"{group[0]}_to_{group[-1]}.html"
        with open(os.path.join(output_directory, fname), "w") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>")
            f.write(html_body)
            f.write("</body></html>")

    # scaffold CSV
    scaffold = pd.DataFrame(index=markers, columns=[f"{sid}_landmark" for sid in sample_ids])
    if implied_landmarks_map is not None:
        for marker in markers:
            for j, sid in enumerate(sample_ids):
                scaffold.at[marker, f"{sid}_landmark"] = implied_landmarks_map[marker][j]
    scaffold.to_csv(os.path.join(output_directory, "landmark_annotations.csv"))

    print(f"✅  All HTML files and landmark_annotations.csv saved to \033[1m'{output_directory}'\033[0m folder")
    