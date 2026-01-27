"""
Model Evaluation Plotting Functions

Comprehensive evaluation plots for Gaussian Neural Network predictions.
These functions assess both prediction accuracy and uncertainty calibration.

Author: CGW
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import torch

# Import existing helper functions for metrics
from DataAnalysis_HelperFcns import mae, rmse, coverage, crps_gaussian


# ==============================================================================
# CORE PERFORMANCE PLOTS
# ==============================================================================

def plot_prediction_vs_truth(y_true, y_pred, sigma_pred,
                              dataset_label='Test Set',
                              show_errorbars=True,
                              color_by_uncertainty=True,
                              figsize=(10, 8)):
    """
    Scatter plot of predicted vs true ages with uncertainty visualization.

    Parameters:
    -----------
    y_true : torch.Tensor or np.ndarray
        True age values
    y_pred : torch.Tensor or np.ndarray
        Predicted age values
    sigma_pred : torch.Tensor or np.ndarray
        Predicted uncertainties (sigma)
    dataset_label : str
        Label for the dataset (e.g., 'Train Set', 'Test Set')
    show_errorbars : bool
        Whether to show error bars (can be cluttered for large datasets)
    color_by_uncertainty : bool
        Color points by uncertainty magnitude
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
    if torch.is_tensor(sigma_pred):
        sigma_pred = sigma_pred.cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=figsize)

    # Compute metrics for title
    mae_val = np.mean(np.abs(y_true - y_pred))
    rmse_val = np.sqrt(np.mean((y_true - y_pred)**2))

    if color_by_uncertainty:
        # Color points by uncertainty
        scatter = ax.scatter(y_true, y_pred, c=sigma_pred,
                            cmap='viridis', alpha=0.6, s=30,
                            label=f'Predictions (N={len(y_true)})')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Uncertainty (σ)', fontsize=12)
    else:
        ax.scatter(y_true, y_pred, alpha=0.5, s=30,
                  label=f'Predictions (N={len(y_true)})')

    # Add error bars for subset if requested (to avoid clutter)
    if show_errorbars:
        # Only show error bars for subset to avoid visual clutter
        n_samples = min(100, len(y_true))
        indices = np.random.choice(len(y_true), n_samples, replace=False)

        ax.errorbar(y_true[indices], y_pred[indices],
                   yerr=2*sigma_pred[indices],  # 2-sigma error bars
                   fmt='none', ecolor='gray', alpha=0.3,
                   capsize=2, label='±2σ error bars (sample)')

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect prediction', zorder=5)

    # Labels and formatting
    ax.set_xlabel('True Age [log(age/Myr)]', fontsize=14)
    ax.set_ylabel('Predicted Age [log(age/Myr)]', fontsize=14)
    ax.set_title(f'{dataset_label}: Predicted vs True Age\n'
                f'MAE = {mae_val:.3f}, RMSE = {rmse_val:.3f}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig, ax


def plot_residuals_analysis(y_true, y_pred, sigma_pred,
                            dataset_label='Test Set',
                            figsize=(14, 10)):
    """
    Comprehensive residual analysis with multiple panels.

    Creates a 2x2 grid:
    - Top left: Residuals vs True Age
    - Top right: Residuals vs Predicted Age
    - Bottom left: Residuals distribution histogram
    - Bottom right: Residuals vs Uncertainty

    Parameters:
    -----------
    y_true : torch.Tensor or np.ndarray
        True age values
    y_pred : torch.Tensor or np.ndarray
        Predicted age values
    sigma_pred : torch.Tensor or np.ndarray
        Predicted uncertainties
    dataset_label : str
        Label for the dataset
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axs : matplotlib figure and axes array
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
    if torch.is_tensor(sigma_pred):
        sigma_pred = sigma_pred.cpu().numpy().flatten()

    # Compute residuals
    residuals = y_pred - y_true

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{dataset_label}: Residual Analysis',
                 fontsize=16, fontweight='bold')

    # -------------------------
    # Panel 1: Residuals vs True Age
    # -------------------------
    ax1 = axs[0, 0]
    ax1.scatter(y_true, residuals, alpha=0.5, s=20)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax1.set_xlabel('True Age [log(age/Myr)]', fontsize=12)
    ax1.set_ylabel('Residuals (Pred - True)', fontsize=12)
    ax1.set_title('Residuals vs True Age', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add running median
    bins = np.linspace(y_true.min(), y_true.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    median_residuals = []
    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            median_residuals.append(np.median(residuals[mask]))
        else:
            median_residuals.append(np.nan)
    ax1.plot(bin_centers, median_residuals, 'g-', linewidth=2,
            label='Running median', marker='o')
    ax1.legend()

    # -------------------------
    # Panel 2: Residuals vs Predicted Age
    # -------------------------
    ax2 = axs[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Predicted Age [log(age/Myr)]', fontsize=12)
    ax2.set_ylabel('Residuals (Pred - True)', fontsize=12)
    ax2.set_title('Residuals vs Predicted Age', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -------------------------
    # Panel 3: Residuals Distribution
    # -------------------------
    ax3 = axs[1, 0]
    ax3.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x, len(residuals) * np.diff(ax3.get_xlim())[0]/30 *
            stats.norm.pdf(x, mu, sigma),
            'r-', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f}²)')

    ax3.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero')
    ax3.axvline(residuals.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean={residuals.mean():.3f}')
    ax3.set_xlabel('Residuals (Pred - True)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'Residuals Distribution (σ={sigma:.3f})', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # -------------------------
    # Panel 4: Residuals vs Uncertainty
    # -------------------------
    ax4 = axs[1, 1]
    abs_residuals = np.abs(residuals)
    ax4.scatter(sigma_pred, abs_residuals, alpha=0.5, s=20)

    # Ideal relationship: |residual| ≈ sigma
    sigma_range = np.linspace(sigma_pred.min(), sigma_pred.max(), 100)
    ax4.plot(sigma_range, sigma_range, 'r--', linewidth=2,
            label='Perfect calibration')

    # Correlation coefficient
    corr = np.corrcoef(sigma_pred, abs_residuals)[0, 1]

    ax4.set_xlabel('Predicted Uncertainty (σ)', fontsize=12)
    ax4.set_ylabel('Absolute Residuals |Pred - True|', fontsize=12)
    ax4.set_title(f'Uncertainty vs Error (corr={corr:.3f})', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axs


def plot_residuals_simple(y_true, y_pred, dataset_label='Test Set', figsize=(12, 5)):
    """
    Simple residual plot: distribution and vs true age.

    Parameters:
    -----------
    y_true : torch.Tensor or np.ndarray
        True age values
    y_pred : torch.Tensor or np.ndarray
        Predicted age values
    dataset_label : str
        Label for the dataset
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axs : matplotlib figure and axes
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()

    residuals = y_pred - y_true

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axs[0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axs[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axs[0].set_xlabel('Residuals (Pred - True)', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)
    axs[0].set_title(f'{dataset_label}: Residuals Distribution', fontsize=12)
    axs[0].grid(True, alpha=0.3, axis='y')

    # Residuals vs True
    axs[1].scatter(y_true, residuals, alpha=0.5, s=20)
    axs[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axs[1].set_xlabel('True Age [log(age/Myr)]', fontsize=12)
    axs[1].set_ylabel('Residuals (Pred - True)', fontsize=12)
    axs[1].set_title(f'{dataset_label}: Residuals vs True Age', fontsize=12)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axs


# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

def compute_summary_metrics(y_true, y_pred, sigma_pred):
    """
    Compute comprehensive summary metrics.

    Parameters:
    -----------
    y_true : torch.Tensor or np.ndarray
        True values
    y_pred : torch.Tensor or np.ndarray
        Predicted values
    sigma_pred : torch.Tensor or np.ndarray
        Predicted uncertainties

    Returns:
    --------
    metrics : dict
        Dictionary of metric names and values
    """
    # Convert to torch if needed
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not torch.is_tensor(sigma_pred):
        sigma_pred = torch.tensor(sigma_pred, dtype=torch.float32)

    # Reshape if needed
    if y_true.dim() == 2:
        y_true = y_true.squeeze()
    if y_pred.dim() == 2:
        y_pred = y_pred.squeeze()
    if sigma_pred.dim() == 2:
        sigma_pred = sigma_pred.squeeze()

    metrics = {
        # Accuracy metrics
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'Median Error': torch.median(torch.abs(y_true - y_pred)).item(),
        'Max Error': torch.max(torch.abs(y_true - y_pred)).item(),

        # Bias
        'Mean Residual': torch.mean(y_pred - y_true).item(),
        'Median Residual': torch.median(y_pred - y_true).item(),

        # Coverage metrics
        'Coverage@68%': coverage(y_true, y_pred, sigma_pred, num_sigma=1.0),
        'Coverage@95%': coverage(y_true, y_pred, sigma_pred, num_sigma=2.0),
        'Coverage@99%': coverage(y_true, y_pred, sigma_pred, num_sigma=3.0),

        # Uncertainty metrics
        'CRPS': crps_gaussian(y_true, y_pred, sigma_pred),
        'Mean Sigma': torch.mean(sigma_pred).item(),
        'Median Sigma': torch.median(sigma_pred).item(),
        'Min Sigma': torch.min(sigma_pred).item(),
        'Max Sigma': torch.max(sigma_pred).item(),

        # Z-score statistics (should be N(0,1) if well-calibrated)
        'Z-score Mean': torch.mean((y_pred - y_true) / sigma_pred).item(),
        'Z-score Std': torch.std((y_pred - y_true) / sigma_pred).item(),
    }

    return metrics


def print_metrics_table(train_metrics, test_metrics):
    """
    Print formatted comparison table of train vs test metrics.

    Parameters:
    -----------
    train_metrics : dict
        Metrics computed on training set
    test_metrics : dict
        Metrics computed on test set
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION METRICS SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Train Set':>15} {'Test Set':>15} {'Difference':>15}")
    print("-"*80)

    for key in train_metrics.keys():
        train_val = train_metrics[key]
        test_val = test_metrics[key]
        diff = test_val - train_val

        print(f"{key:<25} {train_val:>15.4f} {test_val:>15.4f} {diff:>15.4f}")

    print("="*80)
    print("\nInterpretation Guide:")
    print("  • MAE/RMSE: Lower is better (error in log(age/Myr))")
    print("  • Coverage@68%/95%: Should be close to 0.68/0.95 for well-calibrated")
    print("  • CRPS: Lower is better (proper scoring rule)")
    print("  • Z-score Mean: Should be ~0, Std should be ~1 for calibration")
    print("  • If Test >> Train: possible overfitting")
    print("="*80 + "\n")


def plot_metrics_comparison(train_metrics, test_metrics, figsize=(12, 8)):
    """
    Bar plot comparing train vs test metrics.

    Parameters:
    -----------
    train_metrics : dict
        Metrics for training set
    test_metrics : dict
        Metrics for test set
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    # Select metrics to plot (exclude some for clarity)
    metrics_to_plot = ['MAE', 'RMSE', 'CRPS', 'Coverage@68%', 'Coverage@95%',
                       'Median Sigma', 'Z-score Std']

    train_vals = [train_metrics[m] for m in metrics_to_plot]
    test_vals = [test_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Train vs Test Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, ax


# ==============================================================================
# UNCERTAINTY CALIBRATION PLOTS
# ==============================================================================

def plot_zscore_analysis(y_true, y_pred, sigma_pred,
                         dataset_label='Test Set',
                         figsize=(14, 5)):
    """
    Z-score analysis with distribution and QQ-plot.

    A well-calibrated model should have z-scores following N(0,1).

    Parameters:
    -----------
    y_true : torch.Tensor or np.ndarray
        True values
    y_pred : torch.Tensor or np.ndarray
        Predicted values
    sigma_pred : torch.Tensor or np.ndarray
        Predicted uncertainties
    dataset_label : str
        Label for the dataset
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axs : matplotlib figure and axes
    z_scores : np.ndarray
        Computed z-scores
    ks_statistic : float
        KS test statistic
    ks_pvalue : float
        KS test p-value
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
    if torch.is_tensor(sigma_pred):
        sigma_pred = sigma_pred.cpu().numpy().flatten()

    # Compute z-scores: (y_pred - y_true) / sigma_pred
    z_scores = (y_pred - y_true) / sigma_pred

    # Perform KS test against standard normal
    ks_statistic, ks_pvalue = stats.kstest(z_scores, 'norm')

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{dataset_label}: Z-Score Calibration Analysis',
                 fontsize=14, fontweight='bold')

    # -------------------------
    # Panel 1: Z-score histogram
    # -------------------------
    ax1 = axs[0]

    # Histogram
    counts, bins, patches = ax1.hist(z_scores, bins=40, alpha=0.7,
                                     color='steelblue', edgecolor='black',
                                     density=True, label='Observed z-scores')

    # Overlay standard normal
    x = np.linspace(z_scores.min(), z_scores.max(), 200)
    ax1.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
            label='N(0,1) - Ideal')

    # Add statistics
    z_mean = z_scores.mean()
    z_std = z_scores.std()

    ax1.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label='Zero')
    ax1.axvline(z_mean, color='red', linestyle='--', linewidth=2,
               label=f'Mean={z_mean:.3f}')

    ax1.set_xlabel('Z-Score: (Pred - True) / σ', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Z-Score Distribution\n'
                 f'Mean={z_mean:.3f}, Std={z_std:.3f}\n'
                 f'KS test: D={ks_statistic:.4f}, p={ks_pvalue:.4f}',
                 fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -------------------------
    # Panel 2: QQ-plot
    # -------------------------
    ax2 = axs[1]

    # Generate QQ-plot
    stats.probplot(z_scores, dist="norm", plot=ax2)
    ax2.set_title('QQ-Plot vs Normal Distribution', fontsize=12)
    ax2.set_xlabel('Theoretical Quantiles (N(0,1))', fontsize=12)
    ax2.set_ylabel('Sample Quantiles (Z-scores)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add interpretation text
    if ks_pvalue > 0.05:
        interpretation = "✅ GOOD: Cannot reject normality (p>0.05)"
        color = 'green'
    else:
        interpretation = "⚠️ WARNING: Z-scores deviate from N(0,1) (p<0.05)"
        color = 'red'

    ax2.text(0.05, 0.95, interpretation, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()

    return fig, axs, z_scores, ks_statistic, ks_pvalue


def plot_coverage_calibration(y_true, y_pred, sigma_pred,
                               dataset_label='Test Set',
                               figsize=(10, 8)):
    """
    Coverage calibration plot: empirical vs theoretical coverage.

    For a well-calibrated model, empirical coverage should match theoretical
    coverage across all confidence levels.

    Parameters:
    -----------
    y_true : torch.Tensor or np.ndarray
        True values
    y_pred : torch.Tensor or np.ndarray
        Predicted values
    sigma_pred : torch.Tensor or np.ndarray
        Predicted uncertainties
    dataset_label : str
        Label for the dataset
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    coverage_dict : dict
        Dictionary mapping confidence levels to empirical coverage
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
    if torch.is_tensor(sigma_pred):
        sigma_pred = sigma_pred.cpu().numpy().flatten()

    # Define confidence levels to test
    confidence_levels = np.linspace(0.1, 0.99, 30)

    # For Gaussian distribution: confidence_level = erf(num_sigma / sqrt(2))
    # Inverse: num_sigma = sqrt(2) * erf_inv(confidence_level)
    num_sigmas = np.sqrt(2) * stats.erfinv(confidence_levels)

    # Compute empirical coverage at each level
    empirical_coverage = []
    for num_sigma in num_sigmas:
        # Check if true value within predicted interval
        lower_bound = y_pred - num_sigma * sigma_pred
        upper_bound = y_pred + num_sigma * sigma_pred
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        empirical_coverage.append(within_interval.mean())

    empirical_coverage = np.array(empirical_coverage)

    # Create coverage dictionary
    coverage_dict = {
        'confidence_levels': confidence_levels,
        'empirical_coverage': empirical_coverage,
        'num_sigmas': num_sigmas
    }

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot empirical vs theoretical
    ax.plot(confidence_levels, empirical_coverage, 'o-', linewidth=2,
           markersize=4, label='Empirical Coverage', color='steelblue')
    ax.plot(confidence_levels, confidence_levels, 'r--', linewidth=2,
           label='Perfect Calibration', color='red')

    # Highlight common confidence levels
    common_levels = [0.68, 0.95, 0.99]
    for level in common_levels:
        idx = np.argmin(np.abs(confidence_levels - level))
        empirical = empirical_coverage[idx]
        ax.plot(level, empirical, 'go', markersize=10, zorder=5)
        ax.axvline(level, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(empirical, color='gray', linestyle=':', alpha=0.5)

        # Add annotation
        offset = 0.05 if level < 0.9 else -0.1
        ax.annotate(f'{level:.0%}\nEmp: {empirical:.2%}',
                   xy=(level, empirical),
                   xytext=(level + offset, empirical + 0.05),
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', color='black'))

    # Formatting
    ax.set_xlabel('Theoretical Coverage (Confidence Level)', fontsize=14)
    ax.set_ylabel('Empirical Coverage', fontsize=14)
    ax.set_title(f'{dataset_label}: Coverage Calibration Curve\n'
                f'(Points should fall on diagonal for perfect calibration)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Add interpretation text
    # Check if coverage is well-calibrated at common levels
    cov_68_emp = empirical_coverage[np.argmin(np.abs(confidence_levels - 0.68))]
    cov_95_emp = empirical_coverage[np.argmin(np.abs(confidence_levels - 0.95))]

    if abs(cov_68_emp - 0.68) < 0.05 and abs(cov_95_emp - 0.95) < 0.05:
        interpretation = "✅ GOOD: Well-calibrated at 68% and 95%"
        color = 'lightgreen'
    elif cov_68_emp > 0.68 and cov_95_emp > 0.95:
        interpretation = "⚠️ Over-confident (empirical > theoretical)"
        color = 'lightcoral'
    else:
        interpretation = "⚠️ Under-confident (empirical < theoretical)"
        color = 'lightyellow'

    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    plt.tight_layout()

    return fig, ax, coverage_dict


def plot_uncertainty_distribution(sigma_pred_train, sigma_pred_test,
                                   figsize=(12, 5)):
    """
    Plot distribution of predicted uncertainties for train and test sets.

    Parameters:
    -----------
    sigma_pred_train : torch.Tensor or np.ndarray
        Predicted uncertainties for training set
    sigma_pred_test : torch.Tensor or np.ndarray
        Predicted uncertainties for test set
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axs : matplotlib figure and axes
    """
    # Convert to numpy if needed
    if torch.is_tensor(sigma_pred_train):
        sigma_pred_train = sigma_pred_train.cpu().numpy().flatten()
    if torch.is_tensor(sigma_pred_test):
        sigma_pred_test = sigma_pred_test.cpu().numpy().flatten()

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Predicted Uncertainty (σ) Distribution',
                 fontsize=14, fontweight='bold')

    # -------------------------
    # Panel 1: Overlapping histograms
    # -------------------------
    ax1 = axs[0]

    ax1.hist(sigma_pred_train, bins=30, alpha=0.6, color='blue',
            edgecolor='black', label=f'Train (N={len(sigma_pred_train)})')
    ax1.hist(sigma_pred_test, bins=30, alpha=0.6, color='red',
            edgecolor='black', label=f'Test (N={len(sigma_pred_test)})')

    # Add median lines
    ax1.axvline(np.median(sigma_pred_train), color='blue', linestyle='--',
               linewidth=2, label=f'Train median={np.median(sigma_pred_train):.3f}')
    ax1.axvline(np.median(sigma_pred_test), color='red', linestyle='--',
               linewidth=2, label=f'Test median={np.median(sigma_pred_test):.3f}')

    ax1.set_xlabel('Predicted Uncertainty (σ)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Train vs Test Uncertainty', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # -------------------------
    # Panel 2: Box plots
    # -------------------------
    ax2 = axs[1]

    box_data = [sigma_pred_train, sigma_pred_test]
    bp = ax2.boxplot(box_data, labels=['Train', 'Test'],
                     patch_artist=True, widths=0.6)

    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('Predicted Uncertainty (σ)', fontsize=12)
    ax2.set_title('Uncertainty Distribution Summary', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"Train: μ={sigma_pred_train.mean():.3f}, σ={sigma_pred_train.std():.3f}\n"
    stats_text += f"Test:  μ={sigma_pred_test.mean():.3f}, σ={sigma_pred_test.std():.3f}"
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    return fig, axs
