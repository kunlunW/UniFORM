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


# for loading image files (ome-tiff) to dask array
def tifffile_to_dask(im_fp: Union[str, Path]) -> Union[da.array, List[da.Array]]:
    imdata = zarr.open(tifffile.imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = [da.from_zarr(imdata[z]) for z in imdata.array_keys()]
    else:
        imdata = da.from_zarr(imdata)
    return imdata