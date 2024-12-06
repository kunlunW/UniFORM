import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from anndata import AnnData
from .UniFORM_calculate_histogram import (
    preprocess_raw, plot_global_gmm, plot_local_gmm, uniform_calculate_histogram
)
from .UniFORM_landmark_finetuning import (
    plot_distributions_plotly, normalize_and_plot_distributions_plotly
)
from .UniFORM_normalization import (
    adjust_shifted_histograms, plot_distributions, landmark_shift, 
    correlation_based_normalization, normalize_and_plot_distributions, 
    calculate_shift_in_log_pixels, plot_line_histogram, generate_normalized_feature
)

class UniFORM:
    def __init__(self, adata: AnnData):
        """
        Initialize the UniFORM model with an AnnData object.

        Parameters:
        adata (AnnData): Annotated data object to be used in all methods.
        """
        self.adata = adata

    def uniform_calculate_histogram(self, marker_to_plot, **kwargs):
        """
        Wrapper for the `uniform_calculate_histogram` function.
        """
        return uniform_calculate_histogram(self.adata, marker_to_plot, **kwargs)

    def normalize_and_plot_distributions_plotly(self, histograms, gmm_curves, markers_to_plot, output_path, **kwargs):
        """
        Wrapper for the `normalize_and_plot_distributions_plotly` function.
        """
        return normalize_and_plot_distributions_plotly(
            self.adata, histograms, gmm_curves, markers_to_plot, output_path, **kwargs
        )

    def normalize_and_plot_distributions(self, histograms, markers, reference_sample, **kwargs):
        """
        Wrapper for the `normalize_and_plot_distributions` function.
        """
        return normalize_and_plot_distributions(
            self.adata, histograms, markers, reference_sample, **kwargs
        )

    def calculate_shift_in_log_pixels(self, data_range, shifts_fft_dict, **kwargs):
        """
        Wrapper for the `calculate_shift_in_log_pixels` function.
        """
        return calculate_shift_in_log_pixels(data_range, shifts_fft_dict, **kwargs)

    def generate_normalized_feature(self, shift_in_log_pixels_dict, reference_sample, output_directory, **kwargs):
        """
        Wrapper for the `generate_normalized_feature` function.
        """
        return generate_normalized_feature(
            self.adata, shift_in_log_pixels_dict, reference_sample, output_directory, **kwargs
        )
