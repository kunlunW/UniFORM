"""
EDA to determine variations in histograms

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/01/2024

Dependencies:
    - numpy
    - skimage
    - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

def preprocess_raw(image):
    ch = np.log(image)
    ch[ch == -np.inf] = 0
    return ch

def process_samples(crc_dask_arrays, sample_names, marker_list, bin_counts, subplots_per_row, if_grid, tissue_mask=True, tissue_mask_paths=None, xlims=None, ylims=None, save_filename=None):
    results_range = {}
    results_hist = {}
    
    if tissue_mask and (tissue_mask_paths is None or not isinstance(tissue_mask_paths, dict)):
        raise ValueError("tissue_mask is True but tissue_mask_paths is not provided or not a dictionary")
    
    num_markers = len(marker_list)
    rows_needed = np.ceil(num_markers / subplots_per_row).astype(int)
    fig, axes = plt.subplots(rows_needed, subplots_per_row, figsize=(20, rows_needed * 4), squeeze=False)
    axes = axes.flatten()
    
    for marker_index, marker_name in enumerate(marker_list):
        print(f"####################Processing {marker_name}####################")
        min_list = []
        max_list = []
        global_min = float('inf')
        global_max = float('-inf')
        marker_hist_data = []
        
        # First loop: Determine global min and max for the marker across all samples
        for dask_array, sample_name in zip(crc_dask_arrays, sample_names):
            print(f"Processing {sample_name}......")
            IF_dask_array = dask_array[marker_index]
            tile_raw = IF_dask_array.compute()
            tile_scaled = preprocess_raw(tile_raw)
            
            if tissue_mask:
                # need to change for the biolib and ACED dataset
                print(f"loading contour mask for sample {sample_name}")
                HE_tissue_mask_path = tissue_mask_paths.get(sample_name)
                if not HE_tissue_mask_path:
                    raise ValueError(f"Tissue mask path for sample {sample_name} is not provided in tissue_mask_paths")
                
                HE_tissue_mask = imread(HE_tissue_mask_path)
                
                if HE_tissue_mask.shape != tile_scaled.shape:
                    contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                else:
                    contour_mask = HE_tissue_mask   
                    
                tile_scaled_masked = tile_scaled[contour_mask]
                
            else:
                print(f"- Using all pixels")
                tile_scaled_masked = tile_scaled

            min_val = tile_scaled_masked.min()
            max_val = tile_scaled_masked.max()
            min_list.append(min_val)
            max_list.append(max_val)
            
            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)
        
        results_range[marker_name] = {"min_list": min_list, "max_list": max_list, "global_min": global_min, "global_max": global_max}
        
        # Second loop: Plot histograms using the global min and max
        print(f"Plotting the pixel intensidty distribution......")
        hist_list = []
        bin_edge_list = []
        ax = axes[marker_index]
        for dask_array, sample_name in zip(crc_dask_arrays, sample_names):
            IF_dask_array = dask_array[marker_index]
            tile_raw = IF_dask_array.compute()
            tile_scaled = preprocess_raw(tile_raw)
            
            if tissue_mask:
                # need to change for the biolib and ACED dataset
                print(f"loading contour mask for sample {sample_name}")
                HE_tissue_mask_path = tissue_mask_paths.get(sample_name)
                
                # HE_tissue_mask_path = f'{crc_dir}/{sample_name}/{sample_name}_reinhard_mask.tiff' if sample_name != 'CRC01' else f'{crc_dir}/CRC01/CRC01_reinhard_mask.tiff'
                HE_tissue_mask = imread(HE_tissue_mask_path)
                
                if HE_tissue_mask.shape != tile_scaled.shape:
                    contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                else:
                    contour_mask = HE_tissue_mask  
                tile_scaled_masked = tile_scaled[contour_mask]
                
            else:
                tile_scaled_masked = tile_scaled

            hist, bin_edges = np.histogram(tile_scaled_masked, bins=bin_counts, range=(global_min, global_max))
            hist_list.append(hist)
            bin_edge_list.append(bin_edges)
            ax.plot(bin_edges[:-1], hist, label=f'{sample_name}', alpha=0.7)
        
        results_hist[marker_name] = {"hist_list": hist_list, "bin_edge_list": bin_edge_list}
        ax.set_title(f'{marker_name}')
        ax.set_xlabel('Log Scale Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')  # This line ensures every subplot has a legend
        ax.grid(if_grid)
        
        # Set x and y axis limits if specified
        if xlims and marker_index < len(xlims):
            ax.set_xlim(xlims[marker_index])
        if ylims and marker_index < len(ylims):
            ax.set_ylim(ylims[marker_index])
            
    # Hide any unused subplots
    for i in range(num_markers, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename)  # Save the figure to the specified file
        print(f"Figure saved as '{save_filename}' in current working directory")
        
    plt.show()
    
    return results_range, results_hist