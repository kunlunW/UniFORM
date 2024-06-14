import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import os
import warnings
warnings.filterwarnings('ignore')

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


def get_scene_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0].split('_')[-1][:-4]

def process_samples2(if_dask_arrays, sample_names, marker_list, marker_to_plot, bin_counts, subplots_per_row, if_grid, dpi=300, exclude_zero=False, tissue_mask=True, tissue_mask_paths=None, cell_mask=False, cell_mask_paths=None, xlims=None, ylims=None, save_filename=None, sample_dict=None, bl_immune_dict=None):
    """
    Process and plot the intensity distributions of imaging markers from multiple samples, potentially applying a tissue mask.

    Parameters:
    if_dask_arrays (dict of str: list of dask.array.Array): Dictionary where keys are sample names and values are lists of Dask arrays for each sample.
    sample_names (list of str): Names of the samples corresponding to each Dask array.
    marker_list (list of str): List of markers to be processed.
    bin_counts (int): Number of bins for histogram plotting.
    subplots_per_row (int): Number of subplots to be displayed in each row of the resulting plot.
    if_grid (bool): Flag to display grid lines in the plots.
    dpi (int): Dots per inch (resolution) of the plotted figures.
    exclude_zero (bool): If True, zero values are excluded from analysis and plotting.
    tissue_mask (bool): If True, applies a tissue mask provided in `tissue_mask_paths`.
    tissue_mask_paths (dict, optional): Dictionary mapping sample names to lists of file paths of tissue masks.
    cell_mask (bool): If True, applies a cell segmentation mask provided in `cell_mask_paths`.
    cell_mask_paths (dict, optional): Dictionary mapping sample names to lists of file paths of cell segmentation masks.
    xlims (dict, optional): Dictionary mapping marker names to x-axis limits for each subplot.
    ylims (dict, optional): Dictionary mapping marker names to y-axis limits for each subplot.
    save_filename (str, optional): If provided, the plot will be saved to this filename.
    sample_dict (dict of str: list of str, optional): Dictionary where keys are sample names and values are lists of file paths for each sample.
    bl_immune_dict (dict, optional): Dictionary mapping markers to the specific BL-IMMUNE sample to plot.

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

    if cell_mask and (cell_mask_paths is None or not isinstance(cell_mask_paths, dict)):
        raise ValueError("cell_mask is True but cell_mask_paths is not provided or not a dictionary")

    if sample_dict is None or not isinstance(sample_dict, dict):
        raise ValueError("sample_dict is not provided or not a dictionary")

    if bl_immune_dict is None or not isinstance(bl_immune_dict, dict):
        raise ValueError("bl_immune_dict is not provided or not a dictionary")

    biolib_marker_list = [
        'DAPI_R1', 'aSMA', 'Tryp', 'Ki67', 'CD68', 
        'DAPI_R2', 'EPCAM', 'AR', 'CD20', 'ChromA', 
        'DAPI_R3', 'CK5','CD27', 'HLADRB1', 'CD3', 
        'DAPI_R4', 'R4c2', 'CD11b', 'CD4', 'CD45', 
        'DAPI_R5', 'CDX2', 'CD8', 'CD163', 'CD66b', 
        'DAPI_R6', 'p53', 'ERG', 'PD1', 'GZMB', 
        'DAPI_R7', 'FYN', 'Ecad', 'NCAM', 'EOMES', 
        'DAPI_R8', 'Vim', 'NKX31', 'CK8', 'AMACR', 
        'DAPI_R9', 'H3K27ac', 'CD44', 'CD90', 'FOXP3', 
        'DAPI_R10', 'H3K4me27','B7H6', 'FOXA1', 'PTEN'
    ]

    num_markers = sum(1 for marker in marker_list if marker in marker_to_plot)
    rows_needed = np.ceil(num_markers / subplots_per_row).astype(int)
    fig, axes = plt.subplots(rows_needed, subplots_per_row, figsize=(20, rows_needed * 4), dpi=dpi, squeeze=False)
    axes = axes.flatten()

    plotted_marker_index = 0
    
    for marker_index, marker_name in enumerate(marker_list):
        if marker_name not in marker_to_plot:
            continue  # Skip markers not in the marker_to_plot list
        
        print(f"**************************************** Processing {marker_name} ****************************************\n")
        min_list = []
        max_list = []
        global_min = float('inf')
        global_max = float('-inf')
        
        print(f"############################## calculating histogram ##############################")
              
        # First loop: Determine global min and max for the marker across all samples
        for sample_name in sample_names:
            # Skip BL-IMMUNE samples that are not the specified one for this marker
            if sample_name.startswith('BL-IMMUNE') and sample_name != bl_immune_dict.get(marker_name):
                print(f"skipping {sample_name}...")
                continue

            print(f"Processing {sample_name}......")
            combined_data = []

            if sample_name.startswith('BL-IMMUNE'):
                marker_index_biolib = biolib_marker_list.index(marker_name)
                print(f"BL-IMMUNE marker index is: {marker_index_biolib}")
                IF_dask_array = if_dask_arrays[sample_name][0][marker_index_biolib]
                # print(f"IF_dask_array shape is: {IF_dask_array.shape}")
                tile_raw = IF_dask_array.compute()
                tile_scaled = preprocess_raw(tile_raw)

                if exclude_zero:
                    tile_scaled = tile_scaled[tile_scaled > 0]

                if tissue_mask:
                    tissue_mask_path = next((path for path in tissue_mask_paths[sample_name]), None)
                    if not tissue_mask_path:
                        raise ValueError(f"Tissue mask path for sample {sample_name} is not provided in tissue_mask_paths")

                    HE_tissue_mask = imread(tissue_mask_path)

                    if HE_tissue_mask.shape != tile_scaled.shape:
                        contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                    else:
                        contour_mask = HE_tissue_mask

                    tile_scaled = tile_scaled[contour_mask]

                if cell_mask:
                    cell_seg_mask_path = next((path for path in cell_mask_paths[sample_name]), None)
                    if not cell_seg_mask_path:
                        raise ValueError(f"Cell segmentation mask path for sample {sample_name} is not provided in cell_mask_paths")

                    cell_seg_mask = imread(cell_seg_mask_path)

                    if cell_seg_mask.shape != tile_scaled.shape:
                        cell_seg_mask = resize(cell_seg_mask, tile_scaled.shape, order=0)

                    cell_seg_mask_binary = cell_seg_mask != 0
                    tile_scaled = tile_scaled[cell_seg_mask_binary]

                combined_data.append(tile_scaled)

            else:
                for scene_path, dask_array in zip(sample_dict[sample_name], if_dask_arrays[sample_name]):
                    scene_name = get_scene_name(scene_path)
                    IF_dask_array = dask_array[marker_index]
                    # print(f"IF_dask_array shape is: {IF_dask_array.shape}")
                    tile_raw = IF_dask_array.compute()
                    tile_scaled = preprocess_raw(tile_raw)

                    if exclude_zero:
                        tile_scaled = tile_scaled[tile_scaled > 0]

                    if tissue_mask:
                        tissue_mask_path = next((path for path in tissue_mask_paths[sample_name] if scene_name in path), None)
                        if not tissue_mask_path:
                            raise ValueError(f"Tissue mask path for sample {sample_name}, scene {scene_name} is not provided in tissue_mask_paths")

                        HE_tissue_mask = imread(tissue_mask_path)

                        if HE_tissue_mask.shape != tile_scaled.shape:
                            contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                        else:
                            contour_mask = HE_tissue_mask

                        tile_scaled = tile_scaled[contour_mask]

                    if cell_mask:
                        cell_seg_mask_path = next((path for path in cell_mask_paths[sample_name] if scene_name in path), None)
                        if not cell_seg_mask_path:
                            raise ValueError(f"Cell segmentation mask path for sample {sample_name}, scene {scene_name} is not provided in cell_mask_paths")

                        cell_seg_mask = imread(cell_seg_mask_path)

                        if cell_seg_mask.shape != tile_scaled.shape:
                            cell_seg_mask = resize(cell_seg_mask, tile_scaled.shape, order=0)

                        cell_seg_mask_binary = cell_seg_mask != 0
                        tile_scaled = tile_scaled[cell_seg_mask_binary]

                    combined_data.append(tile_scaled)

            # Concatenate combined data for the current sample
            combined_data = np.concatenate(combined_data)

            min_val = combined_data.min()
            max_val = combined_data.max()
            min_list.append(min_val)
            max_list.append(max_val)

            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)

        results_range[marker_name] = {"min_list": min_list, "max_list": max_list, "global_min": global_min, "global_max": global_max}

        # Second loop: Plot histograms using the global min and max
        print(f"############################## calculating histogram ##############################")
        hist_list = []
        bin_edge_list = []

        ax = axes[plotted_marker_index]
        plotted_marker_index += 1

        for sample_name in sample_names:
            # Skip BL-IMMUNE samples that are not the specified one for this marker
            if sample_name.startswith('BL-IMMUNE') and sample_name != bl_immune_dict.get(marker_name):
                print(f"skipping {sample_name}...")
                continue

            print(f"Processing {sample_name}......")
            combined_data = []

            if sample_name.startswith('BL-IMMUNE'):
                marker_index_biolib = biolib_marker_list.index(marker_name)
                IF_dask_array = if_dask_arrays[sample_name][0][marker_index_biolib]
                tile_raw = IF_dask_array.compute()
                tile_scaled = preprocess_raw(tile_raw)

                if exclude_zero:
                    tile_scaled = tile_scaled[tile_scaled > 0]

                if tissue_mask:
                    tissue_mask_path = next((path for path in tissue_mask_paths[sample_name]), None)
                    if not tissue_mask_path:
                        raise ValueError(f"Tissue mask path for sample {sample_name} is not provided in tissue_mask_paths")

                    HE_tissue_mask = imread(tissue_mask_path)

                    if HE_tissue_mask.shape != tile_scaled.shape:
                        contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                    else:
                        contour_mask = HE_tissue_mask

                    tile_scaled = tile_scaled[contour_mask]

                if cell_mask:
                    cell_seg_mask_path = next((path for path in cell_mask_paths[sample_name]), None)
                    if not cell_seg_mask_path:
                        raise ValueError(f"Cell segmentation mask path for sample {sample_name} is not provided in cell_mask_paths")

                    cell_seg_mask = imread(cell_seg_mask_path)

                    if cell_seg_mask.shape != tile_scaled.shape:
                        cell_seg_mask = resize(cell_seg_mask, tile_scaled.shape, order=0)

                    cell_seg_mask_binary = cell_seg_mask != 0
                    tile_scaled = tile_scaled[cell_seg_mask_binary]

                combined_data.append(tile_scaled)

            else:
                for scene_path, dask_array in zip(sample_dict[sample_name], if_dask_arrays[sample_name]):
                    scene_name = get_scene_name(scene_path)
                    IF_dask_array = dask_array[marker_index]
                    tile_raw = IF_dask_array.compute()
                    tile_scaled = preprocess_raw(tile_raw)

                    if exclude_zero:
                        tile_scaled = tile_scaled[tile_scaled > 0]

                    if tissue_mask:
                        tissue_mask_path = next((path for path in tissue_mask_paths[sample_name] if scene_name in path), None)
                        if not tissue_mask_path:
                            raise ValueError(f"Tissue mask path for sample {sample_name}, scene {scene_name} is not provided in tissue_mask_paths")

                        HE_tissue_mask = imread(tissue_mask_path)

                        if HE_tissue_mask.shape != tile_scaled.shape:
                            contour_mask = resize(HE_tissue_mask, tile_scaled.shape, order=0)
                        else:
                            contour_mask = HE_tissue_mask

                        tile_scaled = tile_scaled[contour_mask]

                    if cell_mask:
                        cell_seg_mask_path = next((path for path in cell_mask_paths[sample_name] if scene_name in path), None)
                        if not cell_seg_mask_path:
                            raise ValueError(f"Cell segmentation mask path for sample {sample_name}, scene {scene_name} is not provided in cell_mask_paths")

                        cell_seg_mask = imread(cell_seg_mask_path)

                        if cell_seg_mask.shape != tile_scaled.shape:
                            cell_seg_mask = resize(cell_seg_mask, tile_scaled.shape, order=0)

                        cell_seg_mask_binary = cell_seg_mask != 0
                        tile_scaled = tile_scaled[cell_seg_mask_binary]

                    combined_data.append(tile_scaled)

            # Concatenate combined data for the current sample
            combined_data = np.concatenate(combined_data)

            hist, bin_edges = np.histogram(combined_data, bins=bin_counts, range=(global_min, global_max))
            hist_list.append(hist)
            bin_edge_list.append(bin_edges)
            
            # Highlight BL-IMMUNE sample with thicker line and black color
            linewidth = 2 if sample_name == bl_immune_dict.get(marker_name) else 1
            color = 'black' if sample_name.startswith('BL-IMMUNE') else None
            ax.plot(bin_edges[:-1], hist, label=f'{sample_name}', alpha=0.7, linewidth=linewidth, color=color)

        results_hist[marker_name] = {"hist_list": hist_list, "bin_edge_list": bin_edge_list}
        ax.set_title(f'{marker_name}')
        ax.set_xlabel('Log Scale Pixel Intensity')
        ax.set_ylabel('Frequency')
        # Annotate with the specific BL-IMMUNE sample plotted
        if marker_name in bl_immune_dict:
            # ax.annotate(f"ref: {bl_immune_dict[marker_name]}", xy=(0.5, 0.9), xycoords='axes fraction', fontsize=8, ha='center', color='black')
            ax.annotate(f"ref: {bl_immune_dict[marker_name]}", xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8, ha='left', va='top', color='black')
        ax.grid(if_grid)

        # Set x and y axis limits if specified
        if xlims and marker_name in xlims:
            ax.set_xlim(xlims[marker_name])
        if ylims and marker_name in ylims:
            ax.set_ylim(ylims[marker_name])

    # Hide any unused subplots
    for i in range(plotted_marker_index, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Add a legend outside the subplots, including all BL-IMMUNE samples in black
    # handles, labels = ax.get_legend_handles_labels()
    # unique_labels = set(labels)
    # new_handles = [plt.Line2D([], [], color='black', linewidth=2) for label in unique_labels if label.startswith('BL-IMMUNE')]
    # fig.legend(handles + new_handles, labels + list(unique_labels), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add a legend outside the subplots, including all BL-IMMUNE samples in black
    bl_immune_samples = set(bl_immune_dict.values())
    handles, labels = ax.get_legend_handles_labels()
    bl_immune_handles = [plt.Line2D([], [], color='black', linewidth=2, label=sample) for sample in bl_immune_samples]
    other_handles = [h for h, l in zip(handles, labels) if l not in bl_immune_samples]
    other_labels = [l for l in labels if l not in bl_immune_samples]
    
    fig.legend(bl_immune_handles + other_handles, list(bl_immune_samples) + other_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')  # Save the figure to the specified file
        print(f"Figure saved as '{save_filename}' in current working directory")

    plt.show()

    return results_range, results_hist
