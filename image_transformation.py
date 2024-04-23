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

from plot import plot_gmm, plot_correlations_and_fit_line, plot_line_histogram

def process_and_stack_images(if_dask_arrays, sample_names, marker_dict, shift_in_log_pixels_dict, reference_sample, num_bins, output_directory, dpi=300, segmentation_mask_path=None, plot_img=True, plot_dist=True, plot_single_cell_corr=True, plot_single_cell_img=True, gmm_analysis=True, save_image=True):

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
                print(f"shift scaling factor for {marker} is {np.exp(negated_list[sample_index])}, range is compressed")
                if_marker_raw_uint32 = if_marker_raw.astype(np.uint32) # this will prevent overflow
                if_marker_shifted = if_marker_raw * np.exp(negated_list[sample_index])
                
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
                ax.set_xlim(0, 800)
                
                plt.show()
                
                
            if plot_single_cell_corr: 
                
                # sample_name = f"{sample_names[sample_index]}"
                # mesmer_mask_fname = [f for f in os.listdir(segmentation_mask_path) if sample_name in f]
                # assert len(mesmer_mask_fname) == 1, 'couldnt find segmentation mask, (or found multiple)'
                # mesmer_mask_fname = mesmer_mask_fname[0]
                # cell_mask = imread(f"{segmentation_mask_path}/{mesmer_mask_fname}")
                
                mesmer_mask_fname = segmentation_mask_path[sample_index]
                cell_mask = imread(mesmer_mask_fname)
                
                # sample_name = f"CRC0{sample_index+1}"
                # mesmer_mask_fname = f'{segmentation_mask_dir}/{sample_name.lower()}_mesmer_cell_mask.tif'
                # cell_mask = imread(mesmer_mask_fname)
                
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
            file_path = f"{output_directory}/{sample_names[sample_index]}/{sample_names[sample_index]}_stacked_ref={reference_sample[marker_index]}.tiff"
            tifffile.imwrite(file_path, stacked_images)
            print(f"Stacked TIFF for {sample_names[sample_index]} saved.")
        
    print("##############################Normalization Done##############################\n\n")
    
    
    
def calculate_shift_in_log_pixels(results_range, keys, bin_counts, shifts_fft_dict):
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