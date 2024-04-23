"""
helper functions used for plotting

Author: Mark Kunlun Wang <wangmar@ohsu.edu>
Created: 02/20/2024
Last Modified: 04/19/2024

Dependencies:
    - numpy
    - matplotlib
    - sklearn
    - scipy
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy import stats


def plot_gmm(X, im_name, marker_name, dpi=300, xlims=None):

    print(f"Currently processing {marker_name}......")
    X = np.log(X) 
    X[X == -np.inf] = 0
    
    gmm = GaussianMixture(n_components=2, init_params='random_from_data',max_iter=1000, random_state=10, covariance_type = 'full')
    gmm.fit(X)
    
    predictions = gmm.predict(X)
    class1 = X[predictions == 1]
    class0 = X[predictions == 0]
    
    if not class0.size:
        print("Negative class is empty. Adjust model parameters or input data.")
        class0 = np.array([0])  # Fallback value
    if not class1.size:
        print("Positive class is empty. Adjust model parameters or input data.")
        class1 = np.array([0])  # Fallback value
    
    neg, pos = (class1, class0) if max(class0) > max(class1) else (class0, class1)
    
    x_axis = np.arange(X.min(), X.max(), 0.1)
    y_axis0 = norm.pdf(x_axis, gmm.means_[0], np.sqrt(gmm.covariances_[0])) * gmm.weights_[0] # 1st gaussian
    y_axis1 = norm.pdf(x_axis, gmm.means_[1], np.sqrt(gmm.covariances_[1])) * gmm.weights_[1] # 2nd gaussian
    
    plt.figure(dpi=dpi)
    plt.hist(X, density=True, facecolor='black', alpha=0.7, bins=500)
    plt.title(im_name)
    plt.xlabel(f'{marker_name} mean intensity')
    plt.ylabel('density')
    plt.plot(x_axis, y_axis0[0], c='red')
    plt.plot(x_axis, y_axis1[0],c ='green')
    cut = max(neg)
    plt.axvline(cut, c='blue')
    plt.xlim(xlims)
    plt.show()
    return np.exp(cut)



def plot_correlations_and_fit_line(original_intensity, normalized_intensity, title='Scatter Plot', dpi=300, xlabel='Original Intensity', ylabel='Normalized Intensity'):
    """
    Plot scatter plot of original vs. normalized intensity data, 
    calculate and display Spearman and Pearson correlations, R^2 value, and line of best fit.
    
    Parameters:
    - original_intensity: Array-like, original intensity values.
    - normalized_intensity: Array-like, normalized intensity values.
    - title: String, title of the plot.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    """
    # Calculate Spearman correlation
    spearman_corr, _ = stats.spearmanr(original_intensity, normalized_intensity)

    # Calculate Pearson correlation and R^2 value
    pearson_corr, _ = stats.pearsonr(original_intensity, normalized_intensity)
    r_squared = pearson_corr**2

    # Plotting
    plt.figure(figsize=(5, 5), dpi=dpi)
    plt.scatter(original_intensity, normalized_intensity)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Annotating Spearman correlation and R^2 values
    plt.text(0.05, 0.95, f'Spearman Correlation: {spearman_corr:.5f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'R² Value: {r_squared:.5f}', transform=plt.gca().transAxes)

    # Find the overall min and max points for the line
    pt_min = min(np.min(original_intensity), np.min(normalized_intensity))
    pt_max = max(np.max(original_intensity), np.max(normalized_intensity))

    # Plot the y=x line
    plt.plot([pt_min, pt_max], [pt_min, pt_max], 'b--')

    # Calculate the line of best fit
    slope, intercept, _, _, _ = stats.linregress(original_intensity, normalized_intensity)

    # Generate x values from the min to max observed values
    x_vals = np.linspace(np.min(original_intensity), np.max(original_intensity), 100)

    # Calculate the y values based on the slope and intercept
    y_vals = slope * x_vals + intercept

    # Plot the line of best fit
    plt.plot(x_vals, y_vals, 'r-', label=f'Best Fit Line: y={slope:.2f}x+{intercept:.2f}')

    plt.text(pt_max, pt_max, f'y={slope:.2f}x+{intercept:.2f}', color='red', verticalalignment='bottom', horizontalalignment='right')

    # Display the plot
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_line_histogram(ax, image, label, alpha=0.9, n_bins=1024):
    hist, bins = np.histogram(image.ravel(), bins=n_bins, range=(image.min(), image.max()))
    ax.plot(bins[:-1], hist, label=label, alpha=alpha)
  
    
def plot_distributions(fd, sample_names, t, key, title_suffix, xlim_index, xlim, fig, ax):
    title = f'{key} {title_suffix}'
    for index, sample_name in enumerate(sample_names):
        ax.plot(t, fd.data_matrix[index, :, 0], label=sample_name, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Grid Points')
    ax.set_ylabel('Value')
    if xlim and xlim_index < len(xlim):
        ax.set_xlim(xlim[xlim_index])
    ax.legend(fontsize="large")