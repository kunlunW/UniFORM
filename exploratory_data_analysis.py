"""
EDA to determine variations in histograms

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/19/2024

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
    """
    Apply a logarithmic transformation to an image with clipping handling.

    Parameters:
    image (numpy.ndarray): The input image array where pixel intensities are non-negative.

    Returns:
    numpy.ndarray: An array of the same shape as the input where the logarithm has been applied
                   to all elements. Any -inf values resulting from log(0) are set to zero.
    """
    ch = np.log(image)
    ch[ch == -np.inf] = 0
    return ch

def process_samples(if_dask_arrays, sample_names, marker_list, bin_counts, subplots_per_row, if_grid, dpi=300, exclude_zero=False, tissue_mask=True, tissue_mask_paths=None, xlims=None, ylims=None, save_filename=None):
    """
    Process and plot the intensity distributions of imaging markers from multiple samples, potentially applying a tissue mask.

    Parameters:
    if_dask_arrays (list of dask.array.Array): List of Dask arrays where each array corresponds to a different sample.
    sample_names (list of str): Names of the samples corresponding to each Dask array.
    marker_list (list of str): List of markers to be processed.
    bin_counts (int): Number of bins for histogram plotting.
    subplots_per_row (int): Number of subplots to be displayed in each row of the resulting plot.
    if_grid (bool): Flag to display grid lines in the plots.
    dpi (int): Dots per inch (resolution) of the plotted figures.
    exclude_zero (bool): If True, zero values are excluded from analysis and plotting.
    tissue_mask (bool): If True, applies a tissue mask provided in `tissue_mask_paths`.
    tissue_mask_paths (dict, optional): Dictionary mapping sample names to file paths of tissue masks.
    xlims (list, optional): List of x-axis limits for each subplot.
    ylims (list, optional): List of y-axis limits for each subplot.
    save_filename (str, optional): If provided, the plot will be saved to this filename.

    Returns:
    tuple: A tuple containing two dictionaries:
           - The first dictionary maps each marker to its range of pixel intensities across all samples.
           - The second dictionary maps each marker to histograms of pixel intensities.

    Raises:
    ValueError: If `tissue_mask` is True and `tissue_mask_paths` is not provided or not a dictionary.
    """
    results_range = {}
    results_hist = {}
    
    if tissue_mask and (tissue_mask_paths is None or not isinstance(tissue_mask_paths, dict)):
        raise ValueError("tissue_mask is True but tissue_mask_paths is not provided or not a dictionary")
    
    num_markers = len(marker_list)
    rows_needed = np.ceil(num_markers / subplots_per_row).astype(int)
    fig, axes = plt.subplots(rows_needed, subplots_per_row, figsize=(20, rows_needed * 4), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    
    for marker_index, marker_name in enumerate(marker_list):
        print(f"####################Processing {marker_name}####################")
        min_list = []
        max_list = []
        global_min = float('inf')
        global_max = float('-inf')
        marker_hist_data = []
        
        # First loop: Determine global min and max for the marker across all samples
        for dask_array, sample_name in zip(if_dask_arrays, sample_names):
            print(f"Processing {sample_name}......")
            IF_dask_array = dask_array[marker_index]
            tile_raw = IF_dask_array.compute()
            tile_scaled = preprocess_raw(tile_raw)
            
            if exclude_zero: 
                tile_scaled = tile_scaled[tile_scaled>0] 
            
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
                    
                tile_scaled = tile_scaled[contour_mask]
                
            else:
                print(f"- Using all pixels")
                tile_scaled = tile_scaled

            min_val = tile_scaled.min()
            max_val = tile_scaled.max()
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
        for dask_array, sample_name in zip(if_dask_arrays, sample_names):
            IF_dask_array = dask_array[marker_index]
            tile_raw = IF_dask_array.compute()
            tile_scaled = preprocess_raw(tile_raw)
            
            if exclude_zero: 
                tile_scaled = tile_scaled[tile_scaled>0]
            
            if tissue_mask:
                # need to change for the biolib and ACED dataset
                print(f"loading contour mask for sample {sample_name}")
                HE_tissue_mask_path = tissue_mask_paths.get(sample_name)
                
                HE_tissue_mask = imread(HE_tissue_mask_path)
                
                if HE_tissue_mask.shape != tile_scaled.shape:
                    contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                else:
                    contour_mask = HE_tissue_mask  
                tile_scaled = tile_scaled[contour_mask]
                
            else:
                tile_scaled = tile_scaled

            hist, bin_edges = np.histogram(tile_scaled, bins=bin_counts, range=(global_min, global_max))
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



def process_feature_samples(feature_data, sample_names, marker_list, bin_counts, subplots_per_row, if_grid, dpi=300, xlims=None, ylims=None, save_filename=None):
    """
    Process feature data from multiple samples and plot histograms of cell mean intensities for each marker.

    Parameters:
    feature_data (list of dicts): A list of dictionaries where each dictionary contains 'intensity_mean' key pointing to marker intensity data.
    sample_names (list of str): Names of the samples corresponding to each entry in `feature_data`.
    marker_list (list of str): List of markers to be processed.
    bin_counts (int): Number of bins for histogram plotting.
    subplots_per_row (int): Number of subplots to be displayed in each row.
    if_grid (bool): Flag to display grid lines on the plots.
    dpi (int): Dots per inch (resolution) of the plotted figures.
    xlims (list, optional): X-axis limits for each subplot.
    ylims (list, optional): Y-axis limits for each subplot.
    save_filename (str, optional): Path to save the generated figure.

    Returns:
    tuple: A tuple containing two dictionaries:
           - results_range: maps each marker to its range of pixel intensities across all samples.
           - results_hist: maps each marker to histograms of pixel intensities.
    """
    results_range = {}
    results_hist = {}
    num_markers = len(marker_list)
    rows_needed = np.ceil(num_markers / subplots_per_row).astype(int)
    fig, axes = plt.subplots(rows_needed, subplots_per_row, figsize=(20, rows_needed * 4), dpi=dpi, squeeze=False)
    axes = axes.flatten()

    for marker_index, marker_name in enumerate(marker_list):
        print(f"####################Processing {marker_name}####################")
        min_list = []
        max_list = []
        global_min = float('inf')
        global_max = float('-inf')

        for mean_intensity, sample_name in zip(feature_data, sample_names):
            print(f"Processing {sample_name} for marker {marker_name}......")
            marker_mean_intensity = mean_intensity['intensity_mean'][marker_index]
            marker_mean_intensity_scaled = preprocess_raw(marker_mean_intensity)

            min_val = marker_mean_intensity_scaled.min()
            max_val = marker_mean_intensity_scaled.max()
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
        for mean_intensity, sample_name in zip(feature_data, sample_names):
            marker_mean_intensity = mean_intensity['intensity_mean'][marker_index]
            marker_mean_intensity_scaled = preprocess_raw(marker_mean_intensity)

            hist, bin_edges = np.histogram(marker_mean_intensity_scaled, bins=bin_counts, range=(global_min, global_max))
            hist_list.append(hist)
            bin_edge_list.append(bin_edges)
            ax.plot(bin_edges[:-1], hist, label=f'{sample_name}', alpha=0.7)
        
        results_hist[marker_name] = {"hist_list": hist_list, "bin_edge_list": bin_edge_list}
        ax.set_title(f'{marker_name}')
        ax.set_xlabel('Log Scale Cell Mean Intensity')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')  # This ensures every subplot has a legend
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



def process_samples_per_marker(if_dask_arrays, sample_names, marker_list, bin_counts, if_grid, dpi=300, exclude_zero=False, tissue_mask=True, tissue_mask_paths=None, xlims=None, ylims=None, save_filename=None):
    """
    Process and plot histograms of pixel intensities per marker across multiple samples, optionally applying a tissue mask.

    Parameters:
    if_dask_arrays (list of dask.array.Array): List of Dask arrays, each corresponding to a different sample.
    sample_names (list of str): Names of the samples corresponding to each Dask array.
    marker_list (list of str): List of markers to be processed.
    bin_counts (int): Number of bins for histogram plotting.
    if_grid (bool): Flag to display grid lines on the plots.
    dpi (int): Dots per inch (resolution) for the plots.
    exclude_zero (bool): If True, zero values are excluded from analysis and plotting.
    tissue_mask (bool): If True, applies a tissue mask provided in `tissue_mask_paths`.
    tissue_mask_paths (dict, optional): Dictionary mapping sample names to tissue mask file paths.
    xlims (list, optional): X-axis limits for each subplot.
    ylims (list, optional): Y-axis limits for each subplot.
    save_filename (str, optional): Path to save the generated figure.

    Returns:
    None: This function does not return any values but plots and optionally saves histograms.
    """
    num_markers = len(marker_list)
    num_samples = len(sample_names)
    
    # Iterate over each marker
    for marker_index, marker_name in enumerate(marker_list):
        print(f"####################Processing {marker_name}####################")
        
        # Setup the figure for the current marker with 1 row and num_samples columns
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4), dpi=dpi, squeeze=False)
        
        # Iterate over each sample for the current marker
        for sample_index, (dask_array, sample_name) in enumerate(zip(if_dask_arrays, sample_names)):
            print(f"Processing {sample_name} for marker {marker_name}......")
            IF_dask_array = dask_array[marker_index].compute()  # Compute the slice for the current marker
            tile_scaled = preprocess_raw(IF_dask_array)
            
            if exclude_zero: 
                tile_scaled = tile_scaled[tile_scaled>0] 
            
            if tissue_mask:
                # need to change for the biolib and ACED dataset
                print(f"loading contour mask for sample {sample_name}")
                HE_tissue_mask_path = tissue_mask_paths.get(sample_name)
                
                HE_tissue_mask = imread(HE_tissue_mask_path)
                
                if HE_tissue_mask.shape != tile_scaled.shape:
                    contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                else:
                    contour_mask = HE_tissue_mask  
                tile_scaled = tile_scaled[contour_mask]
                
            else:
                tile_scaled = tile_scaled
            
            # Plot the histogram
            ax = axes[0, sample_index]
            hist, bin_edges = np.histogram(tile_scaled, bins=bin_counts)
            ax.plot(bin_edges[:-1], hist, label=f'{sample_name}', alpha=0.7)
            
            ax.set_title(f'{marker_name}')
            ax.set_xlabel('Log Scale Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
            ax.grid(if_grid)
            
            # Set x and y axis limits if specified
            if xlims and marker_index < len(xlims) and xlims[marker_index]:
                ax.set_xlim(xlims[marker_index])
            if ylims and marker_index < len(ylims) and ylims[marker_index]:
                ax.set_ylim(ylims[marker_index])
        
        plt.tight_layout()
        
        # Optionally save each figure with a specific marker name
        if save_filename:
            marker_specific_filename = f"{save_filename[:-4]}-{marker_name}.png"
            plt.savefig(marker_specific_filename)
            print(f"Figure saved as '{marker_specific_filename}' in current working directory")
        
        plt.show()
        plt.close(fig)  # Close the figure window to free up memory and avoid display clutter
        
        
def process_feature_samples_per_marker(feature_data, sample_names, marker_list, bin_counts, if_grid, dpi=300, xlims=None, ylims=None, save_filename=None):
    """
    Process feature data for each marker across multiple samples and plot histograms of cell mean intensities.

    Parameters:
    feature_data (list of dicts): A list of dictionaries where each dictionary contains 'intensity_mean' key pointing to marker intensity data.
    sample_names (list of str): Names of the samples corresponding to each entry in `feature_data`.
    marker_list (list of str): List of markers to be processed.
    bin_counts (int): Number of bins for histogram plotting.
    if_grid (bool): Flag to display grid lines on the plots.
    dpi (int): Dots per inch (resolution) for the plots.
    xlims (list, optional): X-axis limits for each subplot.
    ylims (list, optional): Y-axis limits for each subplot.
    save_filename (str, optional): Path to save the generated figure with a specific marker name.

    Returns:
    None: This function does not return any values but plots and optionally saves histograms per marker.
    """
    num_markers = len(marker_list)
    num_samples = len(sample_names)
    
    # Iterate over each marker
    for marker_index, marker_name in enumerate(marker_list):
        print(f"####################Processing {marker_name}####################")
        
        # Setup the figure for the current marker with 1 row and num_samples columns
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4), dpi=dpi, squeeze=False)
        
        # Iterate over each sample for the current marker
        for sample_index, (mean_intensity, sample_name) in enumerate(zip(feature_data, sample_names)):
            print(f"Processing {sample_name} for marker {marker_name}......")
            marker_mean_intensity = mean_intensity['intensity_mean'][marker_index]
            marker_mean_intensity_scaled = preprocess_raw(marker_mean_intensity)
            
            # Plot the histogram
            ax = axes[0, sample_index]  # Correctly access the axis object
            hist, bin_edges = np.histogram(marker_mean_intensity_scaled, bins=bin_counts)
            ax.plot(bin_edges[:-1], hist, label=f'{sample_name}', alpha=0.7)
            
            ax.set_title(f'{marker_name}')
            ax.set_xlabel('Log Scale Cell Mean Intensity')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
            ax.grid(if_grid)
            
            # Set x and y axis limits if specified
            if xlims and marker_index < len(xlims) and xlims[marker_index]:
                ax.set_xlim(xlims[marker_index])
            if ylims and marker_index < len(ylims) and ylims[marker_index]:
                ax.set_ylim(ylims[marker_index])
        
        plt.tight_layout()
        
        # Optionally save each figure with a specific marker name
        if save_filename:
            marker_specific_filename = f"{save_filename[:-4]}-{marker_name}.png"
            plt.savefig(marker_specific_filename)
            print(f"Figure saved as '{marker_specific_filename}' in current working directory")
        
        plt.show()
        plt.close(fig)  # Close the figure window to free up memory and avoid display clutter
