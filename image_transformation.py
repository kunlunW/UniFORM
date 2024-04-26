"""
perform image transformation after normalization

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/19/2024

Dependencies:
    - numpy
    - tifffile
    - matplotlib
    - sklearn
    - scipy
    - skimage
    - 
"""

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.io import imread
from skimage.measure import regionprops_table, regionprops
from skimage.transform import resize
import os

from plot import plot_gmm, plot_correlations_and_fit_line, plot_line_histogram
from constants import ome_tiff_dimension_order, ome_tiff_metadata_dict


def calculate_shift_in_log_pixels(results_range, keys, bin_counts, shifts_fft_dict):
    """
    Calculate the shift in log pixels for each marker based on its FFT-derived shifts and intensity range.

    This function processes each specified marker to compute the actual pixel shift values by translating
    the shift values from the frequency domain (obtained via FFT) into log pixel space using the intensity
    range of the marker.

    Parameters:
    results_range (dict): A dictionary mapping each marker to its global intensity range.
    keys (list): List of marker keys that specify which markers to process.
    bin_counts (int): Total number of bins used in the histogram, which defines the resolution of the shift calculation.
    shifts_fft_dict (dict): A dictionary mapping each marker to its list of shifts obtained from FFT analysis.

    Returns:
    dict: A dictionary mapping each marker to its calculated shift in log pixels.

    """
    shift_in_log_pixels_dict = {}
    for key in keys:
        print(f"********** Processing marker {key} **********")
        min_val = results_range[key]['global_min']
        max_val = results_range[key]['global_max']
        increment = (max_val - min_val) / (bin_counts - 1)
        shifts = shifts_fft_dict[key]
        shift_in_log_pixels = [shift * increment for shift in shifts]
        shift_in_log_pixels_dict[key] = shift_in_log_pixels
        print(f"shift_in_log_pixels for {key} is {shift_in_log_pixels}\n")
    return shift_in_log_pixels_dict


def process_and_stack_images(if_dask_arrays, sample_names, marker_dict, shift_in_log_pixels_dict, reference_sample, num_bins, output_directory, dpi=300, segmentation_mask_path=None, plot_img=True, plot_dist=True, plot_single_cell_corr=True, plot_single_cell_img=True, gmm_analysis=True, save_image=True, save_ome_tiff=True):
    
    """
    Process and stack images by normalizing them based on computed log-pixel shifts, and optionally save and plot the results.

    This function processes a collection of images (or channels) for various markers, applies normalization based
    on shifts in log pixels, and generates output for analysis and visualization, including saving the normalized images.

    Parameters:
    if_dask_arrays (list of dask.array.Array): List of Dask arrays, each corresponding to a different sample.
    sample_names (list of str): Names of the samples corresponding to each Dask array.
    marker_dict (list): List of markers to be processed.
    shift_in_log_pixels_dict (dict): Dictionary mapping markers to their respective shift values in log pixels.
    reference_sample (list): List containing reference samples for each marker.
    num_bins (int): Number of bins used in histogramming processes.
    output_directory (str): Directory path where output files will be saved.
    dpi (int): Dots per inch setting for any generated plots.
    segmentation_mask_path (list, optional): Paths to segmentation masks for single cell analysis.
    plot_img (bool): Whether to plot the original and normalized images.
    plot_dist (bool): Whether to plot distribution histograms of pixel intensities.
    plot_single_cell_corr (bool): Whether to plot correlations at the single cell level.
    plot_single_cell_img (bool): Whether to plot images of single cells.
    gmm_analysis (bool): Whether to perform Gaussian Mixture Model analysis.
    save_image (bool): Whether to save the processed images.
    save_ome_tiff (bool): Whether to save images in OME-TIFF format.

    Returns:
    None: This function does not return any values but may output files and plots based on input parameters.
    """

    print("##############################Calculating stretch constant##############################\n")
    # calculating the stretch scaling factor
    # reversing the signs of shifts to fit the pixel domain
    negated_dict = {key: [-x for x in values] for key, values in shift_in_log_pixels_dict.items()}
    
    max_values_dict = {}
    for marker_index, (key, values) in enumerate(negated_dict.items()):
        min_value = min(values)
        min_indices = [i for i, value in enumerate(values) if value == min_value]
        channel_arrays = [if_dask_arrays[i][marker_index] for i in min_indices]    
        max_values_with_indices = [(array.compute().max(), i) for array, i in zip(channel_arrays, min_indices)]
        overall_max, max_index = max(max_values_with_indices, key=lambda x: x[0])
        max_values_dict[key] = overall_max
        print(f"Key index: {marker_index}, Key: {key}, min shift is: {min_value}, Minimum shift sample indices: {min_indices}")
        print(f"max shift sample max value is: {[value for value, index in max_values_with_indices]}")
        print(f"stretch scaling factor for {key} is: {sample_names[max_index]} with a max-value: {overall_max} \n")
        
    print("##############################Done##############################\n\n")
    
    
    print("##############################Performing image normalization##############################\n")
    for sample_index, if_dask_sample in enumerate(if_dask_arrays):
        
        print(f"**********  Processing {sample_names[sample_index]} **********")
        
        processed_images_stack = []
        
        for marker_index, marker in enumerate(marker_dict):
            
            reference_index = sample_names.index(reference_sample[marker_index])
            
            print(f"Processing marker {marker}")
            print(f"Reference sample for {marker} is {reference_sample[marker_index]}")
            
            reference_if_marker_raw = if_dask_arrays[reference_index][marker_dict.index(marker)].compute() # ref marker
            if_marker_raw = if_dask_sample[marker_dict.index(marker)].compute() # each channel 
            
            negated_list = negated_dict[marker]
            
            # if the shift is negative or no shift --> range compression or no shift
            if negated_list[sample_index] <= 0:
                print(f"shift scaling factor for {marker} is {np.exp(negated_list[sample_index])}")
                if_marker_shifted = if_marker_raw * np.exp(negated_list[sample_index])
                
            # if the shift is positive --> range expansion
            if negated_list[sample_index] > 0:
                print(f"shift scaling factor for {marker} is {np.exp(negated_list[sample_index])}, range is expanded")
                if_marker_raw_uint32 = if_marker_raw.astype(np.uint32) # this will prevent overflow
                if_marker_shifted = if_marker_raw_uint32 * np.exp(negated_list[sample_index])
                
            # stretch the raw image for comparison purpose
            if_marker_raw_stretched = ((if_marker_raw - if_marker_raw.min()) / (max_values_dict[marker] - if_marker_raw.min())) * 65535  # Scale to uint16 range
            if_marker_raw_uint16 = np.rint(np.clip(if_marker_raw_stretched, 0, 65535)).astype(np.uint16)
            
            # stretch the normalized image
            if_marker_shifted_stretched = ((if_marker_shifted - if_marker_shifted.min()) / (max_values_dict[marker] - if_marker_shifted.min())) * 65535  # Scale to uint16 range
            if_marker_shifted_uint16 = np.rint(np.clip(if_marker_shifted_stretched, 0, 65535)).astype(np.uint16)
            
            # stretch the reference image
            reference_if_marker_raw_stretched = ((reference_if_marker_raw - reference_if_marker_raw.min()) / (max_values_dict[marker] - reference_if_marker_raw.min())) * 65535  # Scale to uint16 range
            reference_if_marker_raw_uint16 = np.rint(np.clip(reference_if_marker_raw_stretched, 0, 65535)).astype(np.uint16)
            
            # plot the actual original vs normalized images for comparison
            if plot_img: 
                fig, (ax1, ax2) = plt.subplots(1, 2, dpi=dpi, figsize=(10, 5))
                # Processing and plotting the first image
                lower_bound = np.percentile(if_marker_raw_uint16.ravel(), 0.1)
                upper_bound = np.percentile(if_marker_raw_uint16.ravel(), 99.9)
                clipped_img = np.clip(if_marker_raw_uint16, lower_bound, upper_bound)
                clipped_img = clipped_img.astype(np.uint16)
                im1 = ax1.imshow(clipped_img)
                ax1.set_title(f"{sample_names[sample_index]}-{marker}-original-clipped(0.1, 99.9)")
                fig.colorbar(im1, ax=ax1)

                # Processing and plotting the second image
                lower_bound = np.percentile(if_marker_shifted_uint16.ravel(), 0.1)
                upper_bound = np.percentile(if_marker_shifted_uint16.ravel(), 99.9)
                clipped_img = np.clip(if_marker_shifted_uint16, lower_bound, upper_bound)
                clipped_img = clipped_img.astype(np.uint16)
                im2 = ax2.imshow(clipped_img)
                ax2.set_title(f"{sample_names[sample_index]}-{marker}-normalized-clipped(0.1, 99.9)")
                fig.colorbar(im2, ax=ax2)
                
                plt.show()  
                
            # Create a figure with three subplots side by side
            if plot_dist: 
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
                
                plot_line_histogram(ax, if_marker_raw_uint16, f'{sample_names[sample_index]} {marker} - Original', n_bins=num_bins, alpha=0.5)
                plot_line_histogram(ax, if_marker_shifted_uint16, f'{sample_names[sample_index]} {marker} - Normalized', n_bins=num_bins)
                plot_line_histogram(ax, reference_if_marker_raw_uint16, f'{reference_sample[marker_index]} {marker} - Reference', n_bins=num_bins, alpha=0.5)

                ax.set_title(f'{sample_names[sample_index]} {marker} Pixel Intensity Distribution')
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlim(0, 1000)
                
                plt.show()
                
                
            if plot_single_cell_corr: 
                mesmer_mask_fname = segmentation_mask_path[sample_index]
                cell_mask = imread(mesmer_mask_fname)
                cell_mask_resized = resize(cell_mask, if_marker_raw_uint16.shape, order=0)
                original_X_mesmer = regionprops_table(cell_mask_resized, if_marker_raw_uint16, properties=['intensity_mean'])
                normalized_X_mesmer = regionprops_table(cell_mask_resized, if_marker_shifted_uint16, properties=['intensity_mean'])
                
                plot_correlations_and_fit_line(
                    original_X_mesmer['intensity_mean'], 
                    normalized_X_mesmer['intensity_mean'],
                    title=f'Scatter Plot of {sample_names[sample_index]} {marker} original vs normalized',
                    dpi=dpi,
                    xlabel=f'{sample_names[sample_index]} {marker} Original (original scale)',
                    ylabel=f'{sample_names[sample_index]} {marker} Normalized (original scale)'
                )
                
                
                if plot_single_cell_img:
                    # Create a large figure to hold all subplots
                    props = regionprops(cell_mask_resized)
                    
                    fig, axes = plt.subplots(5, 10, figsize=(20, 10), dpi=dpi)
                    axes_flat = axes.flatten()
                    for i, prop in enumerate(props[30000:30050]):
                        minr, minc, maxr, maxc = prop.bbox
                        cell_image = if_marker_raw_uint16[minr:maxr, minc:maxc]
                        ax = axes_flat[i]
                        ax.imshow(cell_image)
                        ax.set_title(f'Cell ID: {prop.label}', fontsize=6)
                        ax.axis('off')  # Hide axes ticks

                    fig.suptitle(f'Random batch of single cells from original {sample_names[sample_index]} {marker}', fontsize=25)
                    plt.tight_layout()
                    plt.show()
                    
                    fig, axes = plt.subplots(5, 10, figsize=(20, 10), dpi=dpi)
                    axes_flat = axes.flatten()
                    for i, prop in enumerate(props[30000:30050]):
                        minr, minc, maxr, maxc = prop.bbox
                        cell_image = if_marker_shifted_uint16[minr:maxr, minc:maxc]
                        ax = axes_flat[i]
                        ax.imshow(cell_image)
                        ax.set_title(f'Cell ID: {prop.label}', fontsize=6)
                        ax.axis('off')  # Hide axes ticks

                    fig.suptitle(f'Random batch of single cells from normalized {sample_names[sample_index]} {marker}', fontsize=25)
                    plt.tight_layout()
                    plt.show()
                    
                
                if gmm_analysis: 
                    original_threshold = plot_gmm(original_X_mesmer['intensity_mean'].reshape(-1, 1), f"{sample_names[sample_index]} {marker} Original", f"{marker} Original", dpi=dpi, xlims=(4,10))
                    condition_met = original_X_mesmer['intensity_mean'] >= original_threshold
                    proportion_ge_threshold = np.mean(condition_met)
                    num_ge_threshold = np.sum(condition_met)
                    print(f"Positive population threshold is: {original_threshold}")
                    print(f"Proportion of positive population >= {original_threshold}: {proportion_ge_threshold*100}%")
                    print(f"Number of positive population >= {original_threshold}: {num_ge_threshold}")
                    
                    normalized_threshold = plot_gmm(normalized_X_mesmer['intensity_mean'].reshape(-1, 1), f"{sample_names[sample_index]} {marker} Normalized", f"{marker} Normalized", dpi=dpi, xlims=(4, 10))
                    condition_met = normalized_X_mesmer['intensity_mean'] >= normalized_threshold
                    proportion_ge_threshold = np.mean(condition_met)
                    num_ge_threshold = np.sum(condition_met)
                    print(f"Positive population threshold is: {normalized_threshold}")
                    print(f"Proportion of elements >= {normalized_threshold}: {proportion_ge_threshold*100}%")
                    print(f"Number of elements >= {normalized_threshold}: {num_ge_threshold}")
                
            if save_image: 
                processed_images_stack.append(if_marker_shifted_uint16)
        
        if save_image:
            stacked_images = np.stack(processed_images_stack, axis=0)
            # check if each sample directory exists or not in the save directory
            folder_path = os.path.join(output_directory, sample_names[sample_index])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
                if save_ome_tiff: 
                    from pyometiff import OMETIFFWriter
                    writer = OMETIFFWriter(
                        fpath=os.path.join(folder_path, f"{sample_names[sample_index]}_normalized.ome.tiff"),
                        dimension_order=dimension_order,
                        metadata=metadata_dict,
                        array=stacked_images,
                        explicit_tiffdata=False)
                    writer.write()
                    
                else: 
                    file_path = os.path.join(folder_path, f"{sample_names[sample_index]}_normalized_stacked.tiff")
                    tifffile.imwrite(file_path, stacked_images)
                    
            print(f"Stacked TIFF for {sample_names[sample_index]} saved.")
        
    print("##############################Normalization Done##############################\n\n")
    

    
def generate_normalized_feature(feature_data, sample_names, marker_dict, shift_in_log_pixels_dict, reference_sample, num_bins, output_directory, dpi=300, plot_dist=True, plot_single_cell_corr=True, gmm_analysis=True, save_normalized_features=True):
    """
    Normalize feature data based on logarithmic pixel shifts and generate various visualizations and analyses.

    This function processes each marker's feature data, normalizes it using calculated shifts, and optionally
    generates histograms, scatter plots, and GMM analysis plots. It also has the capability to save the
    normalized feature data for further analysis.

    Parameters:
    feature_data (list of dicts): List containing dictionaries with 'intensity_mean' keys that hold raw feature data.
    sample_names (list of str): Names of the samples corresponding to each entry in `feature_data`.
    marker_dict (list): List of markers to be processed.
    shift_in_log_pixels_dict (dict): Dictionary mapping markers to their respective shift values in log pixels.
    reference_sample (list): List containing reference samples for each marker.
    num_bins (int): Number of bins used for histogram plotting.
    output_directory (str): Directory path where output files will be saved.
    dpi (int): Dots per inch setting for any generated plots.
    plot_dist (bool): Whether to plot distribution histograms of pixel intensities.
    plot_single_cell_corr (bool): Whether to plot correlations at the single cell level.
    gmm_analysis (bool): Whether to perform Gaussian Mixture Model analysis to determine population thresholds.
    save_normalized_features (bool): Whether to save the processed and normalized feature data.

    Returns:
    None: This function does not return any values but may output files and plots based on input parameters.
    """
    
    negated_dict = {key: [-x for x in values] for key, values in shift_in_log_pixels_dict.items()}
    
    print("##############################Performing feature normalization##############################\n")
            
    for sample_index, mean_intensity in enumerate(feature_data):
        print(f"******************************  Processing {sample_names[sample_index]} ******************************")
        
        normalized_feature = []
        for marker_index, marker in enumerate(marker_dict):
            reference_index = sample_names.index(reference_sample[marker_index])
            print(f"##### Processing marker {marker} #####")
            print(f"Reference sample for {marker} is {reference_sample[marker_index]}")
            reference_marker_raw = feature_data[reference_index]['intensity_mean'][marker_dict.index(marker)] # ref marker
            marker_raw = mean_intensity['intensity_mean'][marker_dict.index(marker)] # each channel 
            
            negated_list = negated_dict[marker]
            
            # if the shift is negative or no shift --> range compression or no shift
            print(f"shift scaling factor for {marker} is {np.exp(negated_list[sample_index])}")
            marker_shifted = marker_raw * np.exp(negated_list[sample_index])
                
            marker_raw = np.clip(marker_raw, 0, 65535)
            marker_shifted = np.clip(marker_shifted, 0, 65535)
            reference_marker_raw = np.clip(reference_marker_raw, 0, 65535)
                
            # Create a figure with three subplots side by side
            if plot_dist: 
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
                plot_line_histogram(ax, marker_raw, f'{sample_names[sample_index]} {marker} - Original', n_bins=num_bins, alpha=0.5)
                plot_line_histogram(ax, marker_shifted, f'{sample_names[sample_index]} {marker} - Normalized', n_bins=num_bins)
                plot_line_histogram(ax, reference_marker_raw, f'{reference_sample[marker_index]} {marker} - Reference', n_bins=num_bins, alpha=0.5)

                ax.set_title(f'{sample_names[sample_index]} {marker} Cell Mean Intensity Distribution')
                ax.set_xlabel('Cell Mean Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlim(0, 1000)
                plt.show()
                
                
            if plot_single_cell_corr: 
                plot_correlations_and_fit_line(
                    marker_raw, 
                    marker_shifted,
                    title=f'Scatter Plot of {sample_names[sample_index]} {marker} original vs normalized',
                    dpi=dpi,
                    xlabel=f'{sample_names[sample_index]} {marker} Original (original scale)',
                    ylabel=f'{sample_names[sample_index]} {marker} Normalized (original scale)'
                )
                 
            if gmm_analysis: 
                original_threshold = plot_gmm(marker_raw.reshape(-1, 1), f"{sample_names[sample_index]} {marker} Original", f"{marker} Original", dpi=dpi, xlims=(4,10))
                condition_met = marker_raw >= original_threshold
                proportion_ge_threshold = np.mean(condition_met)
                num_ge_threshold = np.sum(condition_met)
                print(f"Positive population threshold is: {original_threshold}")
                print(f"Proportion of positive population >= {original_threshold}: {proportion_ge_threshold*100}%")
                print(f"Number of positive population >= {original_threshold}: {num_ge_threshold}")

                normalized_threshold = plot_gmm(marker_shifted.reshape(-1, 1), f"{sample_names[sample_index]} {marker} Normalized", f"{marker} Normalized", dpi=dpi, xlims=(4, 10))
                condition_met = marker_shifted >= normalized_threshold
                proportion_ge_threshold = np.mean(condition_met)
                num_ge_threshold = np.sum(condition_met)
                print(f"Positive population threshold is: {normalized_threshold}")
                print(f"Proportion of elements >= {normalized_threshold}: {proportion_ge_threshold*100}%")
                print(f"Number of elements >= {normalized_threshold}: {num_ge_threshold}")
                
            if save_normalized_features: 
                normalized_feature.append(marker_shifted)
        
        if save_normalized_features:
            stacked_normalized_feature = np.stack(normalized_feature, axis=0)
            stacked_normalized_feature = {'intensity_mean': stacked_normalized_feature}
            # check if each sample directory exists or not in the save directory
            folder_path = os.path.join(output_directory, sample_names[sample_index])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            file_path = os.path.join(folder_path, f"{sample_names[sample_index]}_normalized_feature.pkl")
            with open(file_path, 'wb') as file:
                pickle.dump(stacked_normalized_feature, file)
                    
            print(f"Normalized feature dataset for {sample_names[sample_index]} saved!")
        
    print("##############################Feature Normalization Done##############################\n\n")