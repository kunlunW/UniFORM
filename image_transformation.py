"""
perform image transformation after normalization

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/01/2024

Dependencies:
    - numpy
    - tifffile
"""

import numpy as np
import tifffile


def plot_line_histogram(ax, image, label, alpha=0.9, n_bins=1024):
    hist, bins = np.histogram(image.ravel(), bins=n_bins, range=(image.min(), image.max()))
    ax.plot(bins[:-1], hist, label=label, alpha=alpha)
    
    
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
        print(f"shift_in_log_pixels for {key} is {shift_in_log_pixels}")
    return shift_in_log_pixels_dict

def preprocess_raw(image):
    ch = np.log(image)
    ch[ch == -np.inf] = 0
    return ch

def process_and_stack_images(crc_dask_arrays, marker_dict, shift_in_log_pixels_dict, reference_index, num_bins, plot_dist=True, output_directory):

    for i, crc_dask_sample in enumerate(crc_dask_arrays):
        print(f"**********  Processing CRC0{i+1} **********")
        processed_images_stack = []
        
        for marker in marker_dict:
            print(f"Processing marker {marker}")
            
            # ref marker
            reference_crc_marker_raw = crc_dask_arrays[reference_index][marker_dict.index(marker)].compute()
            
            # raw marker
            CRC_marker_raw = crc_dask_sample[marker_dict.index(marker)].compute()
            CRC_marker_raw_log = preprocess_raw(CRC_marker_raw)
            CRC_marker_raw_log_float64 = CRC_marker_raw_log.astype(np.float64)
            
            negated_list = [-x for x in shift_in_log_pixels_dict[marker]]
            print(f"shift for CRC0{i+1} for {marker} is {negated_list[i]}")
            
            # shifted marker
            CRC_marker_raw_log_shifted = CRC_marker_raw_log_float64 + negated_list[i]
            CRC_marker_raw_log_shifted_original_scale = np.exp(CRC_marker_raw_log_shifted)
            CRC_marker_raw_log_shifted_original_scale_uint16_round = np.rint(CRC_marker_raw_log_shifted_original_scale).astype(np.uint16)
            
            # Create a figure with three subplots side by side
            if plot_dist: 
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                # Plot for CRC02 PD-L1 on ax1
                plot_line_histogram(ax, CRC_marker_raw, f'CRC0{i+1} {marker} - Original', n_bins=num_bins, alpha=0.5)
                plot_line_histogram(ax, CRC_marker_raw_log_shifted_original_scale_uint16_round, f'CRC0{i+1} {marker} - Normalized', n_bins=num_bins)
                plot_line_histogram(ax, reference_crc_marker_raw, f'CRC01 {marker} - Original', n_bins=num_bins, alpha=0.5)

                ax.set_title(f'CRC0{i+1} {marker} Pixel Intensity Distribution')
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlim(0, 800)
            
            processed_images_stack.append(CRC_marker_raw_log_shifted_original_scale_uint16_round)
        
        stacked_images = np.stack(processed_images_stack, axis=0)
        file_path = f"{output_directory}/CRC0{i+1}/CRC0{i+1}_stacked_ref=CRC01.tiff"
        
        tifffile.imwrite(file_path, stacked_images)
        print(f"Stacked TIFF for CRC0{i+1} saved.")
