"""
UniFORM Normalization 

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 12/02/2024

Dependencies:
    - numpy
    - scipy
    - skfda
    - matplotlib
"""

import numpy as np
from scipy.signal import correlate
from scipy import stats
import skfda
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from datetime import datetime

def adjust_shifted_histograms(original_fd, shifted_fd, shifts):
    for i, shift in enumerate(shifts):
        if shift > 0:
            shifted_fd.data_matrix[i][0] = original_fd.data_matrix[i][0]
        elif shift < 0:
            shifted_fd.data_matrix[i][0] = original_fd.data_matrix[i][0]
            shift_to_index = min(-shift, shifted_fd.data_matrix[i].shape[0] - 1)
            shifted_fd.data_matrix[i][shift_to_index] = np.array([0.])


def plot_distributions(fd, sample_names, t, key, title_suffix, xlim_index, xlim, fig, ax, reference_sample, color_map):
    title = f'{key} {title_suffix}'
    for index, sample_name in enumerate(sample_names):
        if sample_name == reference_sample:
            ax.plot(t, fd.data_matrix[index, :, 0], label=sample_name, color='black', linewidth=2.5, alpha=0.7)
        else:
            ax.plot(t, fd.data_matrix[index, :, 0], label=sample_name, color=color_map[sample_name], linewidth=2.5, alpha=0.7)
    
    ax.set_xlabel('Grid Points', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pixel Counts', fontsize=12, fontweight='bold')
    
    if xlim and xlim_index < len(xlim):
        ax.set_xlim(xlim[xlim_index])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.tick_params(axis='both', which='major', width=2)
    ax.legend().remove()

def landmark_shift(fd, landmarks, location): 
    landmarks = np.atleast_1d(landmarks)

    if len(landmarks) != fd.n_samples:
        raise ValueError(
            f"landmark list ({len(landmarks)}) must have the same"
            f" length as the number of samples ({fd.n_samples})",
        )

    # Parses location
    loc_array = np.mean(landmarks) if location is None else np.atleast_1d(location)

    return landmarks - loc_array

def correlation_based_normalization(ref_hist, hist_list):
    shifts_direct = []
    shifts_fft = []

    for hist in hist_list:
        correlation_direct = correlate(hist.flatten(), ref_hist.flatten(), mode='full', method='direct')
        correlation_fft = correlate(hist.flatten(), ref_hist.flatten(), mode='full', method='fft')

        shift_direct = np.argmax(correlation_direct) - (len(ref_hist) - 1)
        shift_fft = np.argmax(correlation_fft) - (len(ref_hist) - 1)

        shifts_direct.append(shift_direct)
        shifts_fft.append(shift_fft)

    return shifts_direct, shifts_fft


def normalize_and_plot_distributions(adata, histograms, markers, reference_sample, landmarks=None,
                                      bin_counts=1024, dpi=300, xlim=None, colormap="tab10", figsize=(13, 5)):
    # Extract sample names dynamically from adata
    sample_names = adata.obs['sample_id'].unique().tolist()

    t = np.linspace(0, bin_counts - 1, bin_counts)
    shifts_fft_dict = {}

    # Create a color palette with a unique color for each sample
    cmap = plt.get_cmap(colormap)
    color_map = {sample_name: cmap(i % cmap.N) for i, sample_name in enumerate(sample_names)}

    for i, marker in tqdm(enumerate(markers), total=len(markers)):
        reference_index = sample_names.index(reference_sample[marker])

        print(f"****************************** Processing marker {marker} ******************************")
        print(f"Reference sample for {marker} is {reference_sample[marker]}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(wspace=0.4)
        
        combined_fd = skfda.FDataGrid(histograms[marker]['hist_list'], sample_points=t, extrapolation="zeros")
        
        plot_distributions(combined_fd, sample_names, t, marker, "Original Distributions", i, xlim, fig, ax1, reference_sample[marker], color_map)

        if landmarks is not None: 
            marker_landmark = landmarks.get(marker, None)
            if marker_landmark is None: 
                print("Performing automatic normalization...")
                _, shift = correlation_based_normalization(combined_fd.data_matrix[reference_index], combined_fd.data_matrix)
            else: 
                print("Performing landmark finetuning...")
                shift = landmark_shift(combined_fd, marker_landmark, marker_landmark[reference_index])
        else:
            print("Performing automatic normalization...")
            _, shift = correlation_based_normalization(combined_fd.data_matrix[reference_index], combined_fd.data_matrix)
        
        shifts_fft_dict[marker] = shift
        # print(f"{marker} shifts is: {shift}\n")
            
        combined_fd_shifted = combined_fd.shift(shift, restrict_domain=False)
        adjust_shifted_histograms(combined_fd, combined_fd_shifted, shift)
        plot_distributions(combined_fd_shifted, sample_names, t, marker, "Normalized Distributions", i, xlim, fig, ax2, reference_sample[marker], color_map)
        
        # Add a legend outside the subplots
        handles, labels = ax1.get_legend_handles_labels()
        ref_handle = plt.Line2D([], [], color='black', linewidth=2, label=f'{reference_sample[marker]} (Reference)')
        other_handles = [h for h, l in zip(handles, labels) if l != reference_sample[marker]]
        other_labels = [l for l in labels if l != reference_sample[marker]]
        fig.legend([ref_handle] + other_handles, [reference_sample[marker]] + other_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

    return shifts_fft_dict



def calculate_shift_in_log_pixels(data_range, shifts_fft_dict, bin_counts=1024):
    """
    Calculate the shift in log pixels for each marker based on its FFT-derived shifts and intensity range.

    This function processes each specified marker to compute the actual pixel shift values by translating
    the shift values from the frequency domain (obtained via FFT) into log pixel space using the intensity
    range of the marker.

    Parameters:
    data_range (dict): A dictionary mapping each marker to its global intensity range.
    bin_counts (int): Total number of bins used in the histogram, which defines the resolution of the shift calculation.
    shifts_fft_dict (dict): A dictionary mapping each marker to its list of shifts obtained from FFT analysis.

    Returns:
    dict: A dictionary mapping each marker to its calculated shift in log pixels.

    """
    markers = list(range_dict.keys())
    shift_in_log_pixels_dict = {}
    
    for key in markers:
        print(f"********** Processing marker {key} **********")
        min_val = data_range[key]['global_min']
        max_val = data_range[key]['global_max']
        increment = (max_val - min_val) / (bin_counts - 1)
        shifts = shifts_fft_dict[key]
        shift_in_log_pixels = [shift * increment for shift in shifts]
        shift_in_log_pixels_dict[key] = shift_in_log_pixels
        print(f"shift_in_log_pixels for {key} is {shift_in_log_pixels}\n")
    return shift_in_log_pixels_dict



def plot_line_histogram(ax, image, label, alpha=0.9, n_bins=1024):
    hist, bins = np.histogram(image.ravel(), bins=n_bins, range=(image.min(), image.max()))
    ax.plot(bins[:-1], hist, label=label, alpha=alpha)


def generate_normalized_feature(
    adata, 
    shift_in_log_pixels_dict, 
    reference_sample, 
    output_directory, 
    num_bins,
    dpi=300, 
    plot_dist=False, 
    save_normalized_features=False
):
    """
    Normalize feature data in anndata and optionally generate visualizations.

    Parameters:
    adata (AnnData): Annotated data object with shape (n_obs, n_vars).
    marker_dict (list): List of markers to be processed.
    shift_in_log_pixels_dict (dict): Dictionary mapping markers to their respective shift values in log pixels.
    reference_sample (list): List containing reference samples for each marker.
    num_bins (int): Number of bins used for histogram plotting.
    output_directory (str): Directory path where the updated AnnData object will be saved.
    dpi (int): Dots per inch setting for any generated plots.
    plot_dist (bool): Whether to plot distribution histograms of pixel intensities.
    save_normalized_features (bool): Whether to save the normalized data to `adata.layers` and save `adata`.

    Returns:
    None
    """
    # Generate a timestamped key for storing normalized features in `adata.layers`
    timestamp = datetime.now().strftime("%Y/%m/%d/%H/%M")
    layer_key = f"{timestamp}_normalized"

    # Negate the shifts for normalization
    negated_dict = {key: [-x for x in values] for key, values in shift_in_log_pixels_dict.items()}

    print("############################## Performing Feature Normalization ##############################\n")

    # Extract feature data and sample names from `adata`
    feature_data = adata.X
    marker_list = adata.var['marker_name'].tolist()
    markers_to_normalize = list(negated_dict.keys())
    sample_names = adata.obs['sample_id'].unique().tolist()

    # Initialize a placeholder for the normalized data
    normalized_data = np.zeros_like(feature_data)

    for marker in tqdm(markers_to_normalize, total=len(markers_to_normalize)):

        print(f"##### Processing marker={marker}, reference={reference_sample[marker]} #####")
        reference_index = sample_names.index(reference_sample[marker])
        marker_index = marker_list.index(marker)

        # Get the shift values for the current marker
        negated_list = negated_dict[marker]

        for sample_index, sample_name in enumerate(sample_names):
            # print(f"Processing sample {sample_name} for marker {marker}")
            
            # Extract raw intensities for the current marker and sample
            marker_raw = feature_data[:, marker_index][adata.obs['sample_id'] == sample_name]
            shift_scaling_factor = 10**(negated_list[sample_index])

            # Apply normalization
            marker_shifted = marker_raw * shift_scaling_factor
            normalized_data[:, marker_index][adata.obs['sample_id'] == sample_name] = marker_shifted

            # Plot histograms if enabled
            if plot_dist:
                reference_marker_raw = feature_data[:, marker_index][adata.obs['sample_id'] == reference_sample[marker]]
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
                plot_line_histogram(ax, np.log10(marker_raw+1), f'{sample_name} {marker} - Original', n_bins=num_bins, alpha=0.5)
                plot_line_histogram(ax, np.log10(marker_shifted+1), f'{sample_name} {marker} - Normalized', n_bins=num_bins)
                plot_line_histogram(ax, np.log10(reference_marker_raw+1), f'{reference_sample[marker]} {marker} - Reference', n_bins=num_bins, alpha=0.5)

                ax.set_title(f'{sample_name} {marker} Cell Mean Intensity Distribution')
                ax.set_xlabel('Log10 Cell Mean Intensity')
                ax.set_ylabel('Frequency')
                ax.legend()
                # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # ax.set_xlim(0, 1500)
                plt.show()

    # Save the normalized data to `adata.layers` with the timestamped key
    adata.layers[layer_key] = normalized_data
    
    if save_normalized_features:

        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save the updated AnnData object
        output_path = os.path.join(output_directory, "normalized_adata.h5ad")
        adata.write(output_path)
        print(f"Normalized data saved in adata.layers under key '{layer_key}'")
        print(f"Updated AnnData object saved at {output_path}")

    print("############################## Feature Normalization Done ##############################\n\n")
