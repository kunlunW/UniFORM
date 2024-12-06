"""
UniFORM Landmark Finetuning Option

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 12/02/2024

Dependencies:
    - numpy
    - plotly
    - matplotlib
    - skfda
"""

import plotly.graph_objects as go
import matplotlib.colors as mcolors
import plotly.io as pio
import plotly.subplots as psub

import numpy as np
import skfda
import matplotlib.pyplot as plt


def plot_distributions_plotly(fd, sample_names, t, key, title_suffix, xlim_index, xlim, fig, row, col, color_map, gmm_curves):
    title = f'{key} {title_suffix}'
    
    fig.update_layout(
    title="",
    title_font=dict(size=30),  # Increase title font size
    xaxis_title='Grid Points',
    xaxis_title_font=dict(size=26),  # Increase x-axis title font size
    yaxis_title='Pixel Counts',
    yaxis_title_font=dict(size=26),  # Increase y-axis title font size
    xaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        tickfont=dict(size=24)  # Increase x-axis ticks label font size
    ),
    yaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        tickfont=dict(size=24)  # Increase y-axis ticks label font size
    ),
    plot_bgcolor='white'
)


    fig.update_layout(title='')

    
    for index, sample_name in enumerate(sample_names):
        color = mcolors.to_hex(color_map[sample_name])
        line_width = 3  # Increased thickness for marker distribution curves
        
        # Plot histogram line
        fig.add_trace(
            go.Scatter(
                x=t,
                y=fd.data_matrix[index, :, 0],
                mode='lines',
                line=dict(color=color, width=line_width),
                name=sample_name,
                text=[f'{sample_name}: ({x}, {y})' for x, y in zip(t, fd.data_matrix[index, :, 0])], 
                opacity=0.7
            ),
            row=row, col=col
        )

        if gmm_curves is None:
            continue  # Skip GMM plotting if gmm_curves is None

        # Convert GMM curves to functional data on the same grid
        gmm_data = gmm_curves[key][sample_name]
        x_axis_fd = skfda.FDataGrid(data_matrix=gmm_data["x_axis"], sample_points=t, extrapolation="zeros")

        # Find histogram peak value for scaling
        hist_max = np.max(fd.data_matrix[index, :, 0])

        # Find peak values in GMM curves
        gmm_max0 = np.max(gmm_data["y_axis0"])
        gmm_max1 = np.max(gmm_data["y_axis1"])

        # Calculate scaling factors
        scale_factor0 = hist_max / gmm_max0 if gmm_max0 > 0 else 1
        scale_factor1 = hist_max / gmm_max1 if gmm_max1 > 0 else 1

        # Scale GMM curves by peak normalization factor
        y_axis0_rescaled = gmm_data["y_axis0"] * scale_factor0
        y_axis1_rescaled = gmm_data["y_axis1"] * scale_factor1
        y_axis0_fd = skfda.FDataGrid(data_matrix=y_axis0_rescaled.reshape(-1), sample_points=t, extrapolation="zeros")
        y_axis1_fd = skfda.FDataGrid(data_matrix=y_axis1_rescaled.reshape(-1), sample_points=t, extrapolation="zeros")

        # Plot the GMM curves as dashed lines with transparency
        fig.add_trace(
            go.Scatter(
                x=x_axis_fd.sample_points[0],
                y=y_axis0_fd.data_matrix[0, :, 0],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
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
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.9,  # Transparency for GMM curve
                showlegend=False
            ),
            row=row, col=col
        )

        # Determine the x-coordinate of the peak of the first Gaussian curve
        gmm_peak_x = gmm_data["x_axis"][np.argmax(gmm_data["y_axis0"])]

        # Add a vertical line at the peak of the first Gaussian curve
        fig.add_trace(
            go.Scatter(
                x=[gmm_peak_x, gmm_peak_x],
                y=[0, hist_max],
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=f"{sample_name} GMM Peak",
                showlegend=False
            ),
            row=row, col=col
        )

    if xlim and xlim_index < len(xlim):
        fig.update_xaxes(range=xlim[xlim_index])

    return fig



def normalize_and_plot_distributions_plotly(adata, histograms, gmm_curves, markers_to_plot, output_path, bin_counts=1024,
                                            dpi=300, xlim=None, colormap='tab10', chunk_size=4):
    # Extract sample names and marker names from the anndata object
    sample_names = adata.obs['sample_id'].unique()

    # Generate color map for samples
    cmap = plt.get_cmap(colormap)
    color_map = {sample_name: cmap(i % 10) for i, sample_name in enumerate(sample_names)}

    # Process samples in chunks
    total_chunks = len(sample_names) // chunk_size + (1 if len(sample_names) % chunk_size != 0 else 0)
    t = np.linspace(0, bin_counts - 1, bin_counts)

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(sample_names))
        current_samples = sample_names[start_idx:end_idx]

        all_html_content = ""

        for i, marker in enumerate(markers_to_plot):
            print(f"Processing marker {marker} for samples {current_samples}")

            # Create a figure for the current marker
            fig = psub.make_subplots(rows=1, cols=1)

            combined_fd = skfda.FDataGrid(
                np.array([histograms[marker]['hist_list'][np.where(sample_names == sample)[0][0]] for sample in current_samples]),
                sample_points=t,
                extrapolation="zeros"
            )

            fig = plot_distributions_plotly(combined_fd, current_samples, t, marker, "Original Distributions", i, xlim, fig, 1, 1, color_map, gmm_curves)

            # Convert figure to HTML
            html_content = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            all_html_content += html_content

        # Save HTML for the current chunk
        output_filename = f"{output_path}/gmm_samples_{start_idx}-{end_idx}.html"
        with open(output_filename, "w") as f:
            f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{all_html_content}</body></html>")
        print(f"Saved HTML for samples {start_idx}-{end_idx} to {output_filename}")
