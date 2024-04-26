"""
helper functions for loading datasets

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/19/2024

Dependencies:
    - tifffile
    - dask
    - zarr
"""

from typing import Union, List
import tifffile
from pathlib import Path
import dask.array as da
import zarr
import os
import pickle
import pandas as pd
import numpy as np


# for loading image files (ome-tiff) to dask array
def tifffile_to_dask(im_fp: Union[str, Path]) -> Union[da.array, List[da.Array]]:
    """
    Load an OME-TIFF file into a Dask array or a list of Dask arrays.

    This function reads an OME-TIFF image file using the `tifffile` library with `aszarr` option,
    which facilitates the lazy loading of image data into a Zarr array. Depending on the
    structure of the OME-TIFF file, this might return a single Dask array or a list of Dask arrays,
    each corresponding to different image planes or channels if the file contains multiple image
    arrays.

    Parameters:
    im_fp (Union[str, Path]): The file path to the OME-TIFF image file.

    Returns:
    Union[da.array, List[da.Array]]: A Dask array or a list of Dask arrays containing the image data.
                                     The data is not loaded into memory until computations are explicitly performed.

    Raises:
    FileNotFoundError: If the file does not exist at the specified path.
    ValueError: If the file is not in a supported OME-TIFF format.
    """
    
    imdata = zarr.open(tifffile.imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = [da.from_zarr(imdata[z]) for z in imdata.array_keys()]
    else:
        imdata = da.from_zarr(imdata)
    return imdata

def universal_feature_data_loader(file_path):
    """
    Load feature data from a file into a suitable data structure based on the file extension.

    This function supports loading data from various file formats including NumPy arrays (.npy),
    CSV files (.csv), and Python pickle files (.pkl or .pickle). It automatically determines the
    appropriate method to load the data based on the file extension.

    Parameters:
    file_path (str): The path to the file from which to load the data.

    Returns:
    Various: Depending on the file type, this function returns:
             - NumPy array for .npy files
             - pandas DataFrame for .csv files
             - Any Python object for pickle files

    Raises:
    ValueError: If the file extension is not supported.
    FileNotFoundError: If the file does not exist at the specified path.
    Exception: For any other errors encountered during file loading.
    """
    # Get the file extension
    _, file_ext = os.path.splitext(file_path)

    # Load data according to the file type
    if file_ext in ['.npy']:  # NumPy file
        data = np.load(file_path)
        return data

    elif file_ext in ['.csv']:  # CSV file
        data = pd.read_csv(file_path)
        return data

    elif file_ext in ['.pkl', '.pickle']:  # Pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    else:
        raise ValueError("Unsupported file type")