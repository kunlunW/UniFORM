"""
Normalize and Plot Distributions

This script contains functions to normalize and plot distribution data for comparison. 
It utilizes correlation-based normalization, calculates correlations and divergences, 
and adjusts shifted histograms.

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/19/2024

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
from plot import plot_distributions


def correlation_based_normalization(ref_hist, hist_list):
    """
    Calculate histogram shifts using both direct and FFT-based correlation methods.

    Parameters:
    ref_hist (array_like): The reference histogram array.
    hist_list (list of array_like): A list of histograms to be normalized relative to the reference histogram.

    Returns:
    tuple: A tuple containing four elements:
        - shifts_direct (list of int): List of shift values calculated using direct correlation.
        - shifts_fft (list of int): List of shift values calculated using FFT-based correlation.
        - correlation_list_direct (list of array_like): List of correlation results using direct method.
        - correlation_list_fft (list of array_like): List of correlation results using FFT method.
    """
    shifts_direct = []
    shifts_fft = []
    
    correlation_list_direct = []
    correlation_list_fft = []
    for hist in hist_list:
        # Calculate correlation with the reference histogram
        correlation_direct = correlate(hist.flatten(), ref_hist.flatten(), mode='full', method='direct')
        correlation_fft = correlate(hist.flatten(), ref_hist.flatten(), mode='full', method='fft')
        
        correlation_list_direct.append(correlation_direct)
        correlation_list_fft.append(correlation_fft)
        
        # Find the maximum correlation index
        shift_direct = np.argmax(correlation_direct) - (len(ref_hist) - 1)
        shift_fft = np.argmax(correlation_fft) - (len(ref_hist) - 1)
        
        # Add to the list of normalized histograms
        shifts_direct.append(shift_direct)
        shifts_fft.append(shift_fft)
        
    return shifts_direct, shifts_fft, correlation_list_direct, correlation_list_fft


def calculate_correlations_and_divergences(combined_fd):
    """
    Calculates Pearson and Spearman correlations, and Kullback-Leibler divergence for histograms.

    This function processes a set of histograms (functional data) and computes the Pearson and Spearman
    correlations as well as the KL divergence against the reference histogram.

    Parameters:
    combined_fd (FDataGrid): A functional data grid object containing the histograms and their bin edges.

    Returns:
    dict: A dictionary with three keys: 'pearson_correlations', 'spearman_correlations', and 'kl_divergences'.
          Each key maps to a list of the corresponding values calculated against the reference histogram.
    """
    # Calculate Pearson and Spearman correlations for each histogram against the reference
    pearson_correlations = []
    spearman_correlations = []

    for i in range(combined_fd.data_matrix.shape[0]):
        pearson_corr, _ = stats.pearsonr(combined_fd.data_matrix[0].flatten(), combined_fd.data_matrix[i].flatten())
        spearman_corr, _ = stats.spearmanr(combined_fd.data_matrix[0].flatten(), combined_fd.data_matrix[i].flatten())
        
        pearson_correlations.append(f"{pearson_corr:.5g}")
        spearman_correlations.append(f"{spearman_corr:.5g}")

    def kl_divergence(P, Q):
        """Calculate the KL Divergence between two histograms."""
        epsilon = 1e-10  # Small constant to avoid division by zero or log(0)
        P = P + epsilon
        Q = Q + epsilon
        return np.sum(P * np.log(P / Q))

    histograms = combined_fd.data_matrix / combined_fd.data_matrix.sum(axis=1, keepdims=True)

    # Calculate KL Divergence for each histogram against the reference
    kl_divergences = []

    for i in range(histograms.shape[0]):
        kl_div = kl_divergence(histograms[0], histograms[i])
        kl_divergences.append(f"{kl_div:.5g}")

    return {
        "pearson_correlations": pearson_correlations,
        "spearman_correlations": spearman_correlations,
        "kl_divergences": kl_divergences
    }



def adjust_shifted_histograms(original_fd, shifted_fd, shifts):
    """
    Adjust the first bin of shifted histograms to correct for artifacts introduced by shifting operations.

    Parameters:
    original_fd (FDataGrid): The original functional data object containing unshifted histograms.
    shifted_fd (FDataGrid): The functional data object containing histograms after applying shifts.
    shifts (list of int): A list of integers representing the shift for each histogram. Positive values
                          indicate a leftward shift, and negative values indicate a rightward shift.

    Returns:
    None: The function modifies the shifted_fd data_matrix in place to realign histogram bins and does not return any value.
    """
    for i, shift in enumerate(shifts):
        if shift > 0:
            # For positive shifts, realign the first bin and pad the end with zeros.
            shifted_fd.data_matrix[i][0] = original_fd.data_matrix[i][0]
        elif shift < 0:
            # For negative shifts, realign the first bin and zero out the shifted bin.
            shifted_fd.data_matrix[i][0] = original_fd.data_matrix[i][0]
            # Ensure the shifted-to index does not exceed the histogram length
            shift_to_index = min(-shift, shifted_fd.data_matrix[i].shape[0] - 1)
            shifted_fd.data_matrix[i][shift_to_index] = np.array([0.])
            
            
def normalize_and_plot_distributions(results_hist, keys, sample_names, reference_sample, bin_counts, dpi=300, xlim=None):
    """
    Normalize histograms based on correlation-based shifts and plot both original and normalized distributions.

    This function normalizes histograms for various markers using shifts determined by correlation and plots
    the results, displaying the changes and statistical metrics before and after normalization.

    Parameters:
    results_hist (dict): Dictionary mapping each marker to its histogram data.
    keys (list of str): List of marker keys specifying which histograms to normalize.
    sample_names (list of str): List of sample names corresponding to each histogram.
    reference_sample (list of str): List specifying reference samples for each marker.
    bin_counts (int): The number of bins in each histogram.
    dpi (int): Dots per inch resolution for the generated plots.
    xlim (tuple, optional): X-axis limits for plotting histograms.

    Returns:
    dict: A dictionary mapping each marker to its list of shift values determined by FFT.
    """
    # reference_index = sample_names.index(reference_sample)
    t = np.linspace(0, bin_counts-1, bin_counts)  # Assuming this is constant for all histograms
    shifts_fft_dict = {}

    for i, key in enumerate(keys):
        reference_index = sample_names.index(reference_sample[i])
    
        print(f"****************************** Processing marker {key} ******************************")
        print(f"Reference sample for {key} is {reference_sample[i]}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), dpi=dpi)

        # original distributions
        combined_fd = skfda.FDataGrid(results_hist[key]['hist_list'], sample_points=t, extrapolation="zeros")
        metrics_results = calculate_correlations_and_divergences(combined_fd)
        # plot_distributions(combined_fd, sample_names, t, key, "Original Distributions", i, xlim)
        plot_distributions(combined_fd, sample_names, t, key, "Original Distributions", i, xlim, fig, ax1)
        
        # normalized/aligned distributions
        shifts_direct, shifts_fft, _, _ = correlation_based_normalization(combined_fd.data_matrix[reference_index], combined_fd.data_matrix)
        shifts_fft_dict[key] = shifts_fft
        
        combined_fd_shifted = combined_fd.shift(shifts_fft, restrict_domain=False)    
        adjust_shifted_histograms(combined_fd, combined_fd_shifted, shifts_fft)
        metrics_results_shifted = calculate_correlations_and_divergences(combined_fd_shifted)
        # plot_distributions(combined_fd_shifted, sample_names, t, key, "Normalized Distributions", i, xlim)
        plot_distributions(combined_fd_shifted, sample_names, t, key, "Normalized Distributions", i, xlim, fig, ax2)
        plt.show() 
        
        print(f"##### Metrics for original marker image #####")
        print(f"{key} Pearson Correlation Coefficients:", metrics_results['pearson_correlations'])
        print(f"{key} Spearman Correlation Coefficients:", metrics_results['spearman_correlations'])
        print(f"{key} KL Divergences:", metrics_results['kl_divergences'], "\n")
        
        print(f"##### Metrics for shifted marker image #####")
        print(f"{key} shifted Pearson Correlation Coefficients:", metrics_results_shifted['pearson_correlations'])
        print(f"{key} shifted Spearman Correlation Coefficients:", metrics_results_shifted['spearman_correlations'])
        print(f"{key} shifted KL Divergences:", metrics_results_shifted['kl_divergences'], "\n")
    
        print(f"{key} shifts_direct is: {shifts_direct}")
        print(f"{key} shifts_fft is:    {shifts_fft}\n")
        
    return shifts_fft_dict