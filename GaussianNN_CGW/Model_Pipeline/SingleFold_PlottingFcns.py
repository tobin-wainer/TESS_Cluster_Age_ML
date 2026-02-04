import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.model_selection import ParameterGrid
import matplotlib.gridspec as gridspec


"""
These are plotting functions which require inputs of a dictionary, and some additional info. 
The dictionary sturcture can be seen in the training notebooks, but quickly it is stuctured as:
The dictionary is mostly sturcutred like an array, with 

all_runs_log = {
    model_name: {
        run_id: {
            **{
                "params": config_dict,
            },
            **{
                k_id: {
                    "model": None,
                    "log": copy.deepcopy(empty_train_log)
                } for k_id in range(config_dict['num_kfolds'])
            }
        } for run_id, config_dict in enumerate(ParameterGrid(params))
    } for model_name in model_names
}
    
NOTE: the code in this notebook was written before the k-fold structure, so will not always work with it.
I tried to update the functions to add that functionality but some fcns that are less useful I didn't get to.

I would in most cases recommend writing your own visualization code as this is fairly old.
"""


def plot_Ypred_v_Ytrue_withErrs(all_runs_log, model_names, model_data):
    x_data, period_data, y_data = model_data
    for model_idx, model_name in enumerate(model_names):
        runs = all_runs_log[model_name]
        num_runs = len(runs)
        cols = 3
        rows = int(np.ceil(num_runs / cols))
        
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        fig.suptitle(f"{model_name} - y_true vs y_pred with errors", fontsize=16)
        
        # Pull the relevant data for this model
        #X_train, X_test, y_train, y_test, X_test_BDATA, y_test_BDATA, ATTRIBUTES, SCALER = model_data[model_idx]

        X = x_data[model_idx]
        X_period = period_data[model_idx]
        Y = y_data[model_idx].numpy().flatten()
        #X = X_test
        #Y = y_test.numpy().flatten()

        for run_idx, (run_id, run_info) in enumerate(runs.items()):
            print(run_idx)
            row, col = divmod(run_idx, cols)
            ax = axs[row][col]

            model = run_info["model"]
            config = run_info["params"]
            #if (config['lr']==0.0001) and (config['batch_size']==100):
            if (run_idx < 10):
                
                if model is None:
                    ax.set_title(f"{run_id}: (No model)")
                    ax.axis("off")
                    continue
    
                model.eval()
                with torch.no_grad():
                    y_pred, sigma_pred = model(X, X_period)
                    y_pred = y_pred.numpy().flatten()
                    sigma_pred = sigma_pred.numpy().flatten()
    
                # Calculate statistics
                abs_error = np.abs(y_pred - Y)
                median_sigma = np.median(sigma_pred)
                median_abs_error = np.median(abs_error)
    
                ax.scatter(Y, y_pred, s=2, color='blue', alpha=0.6)
                ax.errorbar(Y, y_pred, yerr=sigma_pred, fmt='o', color='orange', alpha=0.05, markersize=2)
    
                min_val = min(Y.min(), y_pred.min())
                max_val = max(Y.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)  # 1-to-1 line
    
                # Split config dictionary into multiple lines evenly across 3 lines
                config_lines = [f"{key}: {value}" for key, value in config.items()]
                num_config_lines = len(config_lines)
                lines_per_chunk = np.ceil(num_config_lines / 3)  # Number of items per line
                # Split config into chunks and format them into 3 lines
                config_chunks = [config_lines[i:i + int(lines_per_chunk)] for i in range(0, num_config_lines, int(lines_per_chunk))]
                config_text = "\n".join(["\n".join(chunk) for chunk in config_chunks])  # Join the chunks

                ax.set_title(
                    f"Run {run_id}\n"
                    f"median σ={median_sigma:.3f}, median |err|={median_abs_error:.3f}\n"
                    f"{config_text}",
                    fontsize=8
                )
                ax.set_xlabel("y_true")
                ax.set_ylabel("y_pred")
                ax.set_ylim([-1.5, 2])
                ax.set_xlim([-1.5, 2])

    # Hide any unused subplots
    for i in range(num_runs, rows * cols):
        row, col = divmod(i, cols)
        axs[row][col].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
'''
Example Usage:
data = (model_train_data_statistics, model_train_data_periodogram, model_train_data_Y)
plot_Ypred_v_Ytrue_withErrs(all_runs_log, model_names, data)
'''






#######################################################################################################################################
#######################################################################################################################################

import numpy as np
from scipy.stats import norm, kstest
import scipy

##### TEST METRICS ###

# --- Accuracy Metrics --- These test how close the predictions are to the true values (ignoring σ):

def mae(y_true, y_pred):
    # Mean Absolute Error
    # Robust to Outliers
    
    return torch.mean(torch.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    # Root Mean Squared Error
    # Sensitive to large errors — shows how “bad” outliers are
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mean_normalized_error(y_true, y_pred, sigma):
    # This is useful to check if errors are consistent with predicted uncertainties
    return torch.mean(torch.abs((y_true - y_pred) / sigma))

# --- Uncertainty Calibration --- These test whether your predicted σ values are meaningful:
def coverage_probability(y_true, y_pred, sigma, factor=1.0):
    # For a 1σ prediction interval (68.3%), check what fraction of targets fall within:
    # If the model is well-calibrated, ~68% should fall in the band.
    return torch.mean((torch.abs(y_true - y_pred) <= factor * sigma).float())

def z_scores(y_true, y_pred, sigma):
    return (y_true - y_pred) / sigma

def z_score_ks_test(y_true, y_pred, sigma):
    # Computes z-scores and returns the KS test statistic and p-value
    z = z_scores(y_true, y_pred, sigma).cpu().numpy()
    return kstest(z, 'norm')  # returns (D-statistic, p-value)

def skewness(x):
    mean = torch.mean(x)
    std = torch.std(x, unbiased=False)
    return torch.mean(((x - mean) / std) ** 3)

def kurtosis(x):
    mean = torch.mean(x)
    std = torch.std(x, unbiased=False)
    return torch.mean(((x - mean) / std) ** 4)

# --- Probabilistic Score --- These test whether your predicted σ values are meaningful:
def negative_log_likelihood(y_true, y_pred, sigma):
    return torch.mean(0.5 * ((y_true - y_pred) ** 2 / sigma ** 2 + torch.log(sigma ** 2)))

# --- Other ---
def median_sigma2(sigma):
    return torch.median(sigma)

def mean_sigma2(sigma):
    return torch.mean(sigma)

def min_sigma2(sigma):
    return torch.min(sigma)

def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, np.ndarray):
        return x.flatten()[0]  # works even for (1,) or (1,1)
    else:
        return float(x)

def calculate_MCDropout_ZScoreDistribution(model, X, X_period, y_true):

    return

def SummaryStats(model, X, X_period, y_true):
    """
    Compute and print summary statistics for model performance.
    
    Inputs:
        model   : PyTorch model that outputs (y_pred, sigma)
        X_test  : input tensor
        y_true  : ground truth tensor
    """
    model.eval() 
    with torch.no_grad():
        y_pred, sigma = model(X, X_period)
    
    # Accuracy metrics
    MAE = mae(y_true, y_pred)
    RMSE = rmse(y_true, y_pred)
    MNE = mean_normalized_error(y_true, y_pred, sigma)

    # Uncertainty calibration
    CovProb = coverage_probability(y_true, y_pred, sigma, factor=1.0)
    Zscores = z_scores(y_true, y_pred, sigma)
    KsStat, KsPval = z_score_ks_test(y_true, y_pred, sigma)
    Zscores_std = np.std(Zscores.cpu().numpy()) #ideally ≈ 1
    Zscores_skew = skewness(Zscores)
    Zscores_kurt = kurtosis(Zscores)

    # Ensure scalars
    CovProb = CovProb.item()
    KsStat = KsStat.item() if isinstance(KsStat, torch.Tensor) else float(KsStat)
    KsPval = KsPval.item() if isinstance(KsPval, torch.Tensor) else float(KsPval)

    # Probabilistic loss
    NLL = negative_log_likelihood(y_true, y_pred, sigma)

    # Sigma stats
    med_sig = median_sigma2(sigma)
    mean_sig = mean_sigma2(sigma)

    print("="*50)
    print("Summary Statistics")
    print("="*50)

    print("🔍 Accuracy Metrics:")
    print(f"   • Mean Absolute Error     : {MAE:.4f}")
    print(f"   • Root Mean Square Error  : {RMSE:.4f}")
    print(f"   • Mean Normalized Error   : {MNE:.4f}")
    print()

    print("📏 Uncertainty Calibration:")
    print(f"   • Coverage (|err| < 1σ)   : {CovProb:.4f}")
    print(f"   • KS Test (z ~ N(0,1))    : stat={KsStat:.4f}, p={KsPval:.4f}")
    print(f"   • Zscores STD (~=1?)      : stat={Zscores_std:.4f}")
    print(f"   • Zscores Skewness (~=0?)      : stat={Zscores_skew:.4f}")
    print(f"   • Zscores Kurtosis (~=3?)      : stat={Zscores_kurt:.4f}")
    print()
    
    print("📉 Probabilistic Performance:")
    print(f"   • Negative Log Likelihood : {NLL:.4f}")
    print()

    print("📊 Predicted Sigma Summary:")
    print(f"   • Median σ                : {med_sig:.4f}")
    print(f"   • Mean σ                  : {mean_sig:.4f}")
    print("="*50)

    return {
        'MAE': to_scalar(MAE),
        'RMSE': to_scalar(RMSE),
        'MNE': to_scalar(MNE),
        'Coverage': to_scalar(CovProb),
        'Zscore_Std': to_scalar(Zscores_std),
        'Zscore_Skew': to_scalar(Zscores_skew),
        'Zscore_Kurt': to_scalar(Zscores_kurt),
        'KS_stat': to_scalar(KsStat),
        'KS_pval': to_scalar(KsPval),
        'NLL': to_scalar(NLL),
        'Median_sigma': to_scalar(med_sig),
        'Mean_sigma': to_scalar(mean_sig),
        'Zscores': Zscores.cpu().numpy(),
        'Sigmas': sigma.cpu().numpy()

    }
    # ADD IN KURTOSIS!
    

def PlotSummaryStats(results_list, model_names, figsize=(15, 40)):
    names = np.array(model_names)
    n_models = len(model_names)

    # Extract metrics
    MAEs = np.array([res['MAE'] for res in results_list])
    RMSEs = np.array([res['RMSE'] for res in results_list])
    MNEs = np.array([res['MNE'] for res in results_list])
    Coverage = np.array([res['Coverage'] for res in results_list])
    Zscore_Std = np.array([res['Zscore_Std'] for res in results_list])
    Zscore_Skew = np.array([res['Zscore_Skew'] for res in results_list])
    Zscore_Kurt = np.array([res['Zscore_Kurt'] for res in results_list])
    KS_stat = np.array([res['KS_stat'] for res in results_list])
    KS_pvals = np.array([res['KS_pval'] for res in results_list])
    NLLs = np.array([res['NLL'] for res in results_list])
    MedSig = np.array([res['Median_sigma'] for res in results_list])
    MeanSig = np.array([res['Mean_sigma'] for res in results_list])
    ZscoreSets = [res.get('Zscores', None) for res in results_list]
    SigmasSets = [res.get('Sigmas', None) for res in results_list]

    # All metrics
    metrics = {
        "Mean Absolute Err": MAEs,
        "RMSE": RMSEs,
        "Mean Normalized Err": MNEs,
        "Coverage (|err| < 1σ)": Coverage,
        "Zscore STD (~=1?)": Zscore_Std,
        "Zscore Skew+1 (~=1?)": Zscore_Skew+1,
        "Zscore Kurtosis (~=3?)": Zscore_Kurt,
        "KS statistic (z ~ N(0,1))": KS_stat,
        "KS p-value": KS_pvals,
        "NLL": NLLs,
        "Median σ": MedSig,
        "Mean σ": MeanSig
    }

    metric_names = list(metrics.keys())
    num_metrics = len(metric_names)

    # Plotting
    fig, (ax_bars, ax_hist_z, ax_hist_sigma, ax_hist_sigma2, ax_hist_zCDF) = plt.subplots(5, 1, figsize=figsize, 
                                                                                          gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

    x = np.arange(num_metrics)
    width = 0.8 / n_models  # total group width = 0.8

    colors = plt.cm.tab10.colors  # Up to 10 unique colors
    for i, model_name in enumerate(model_names):
        vals = [metrics[m][i] for m in metric_names]
        ax_bars.bar(x + i * width, vals, width=width, label=model_name, color=colors[i % len(colors)])

    ax_bars.set_xticks(x + width * (n_models - 1) / 2)
    ax_bars.set_xticklabels(metric_names, rotation=45, ha='right')
    ax_bars.set_title("Summary Statistics by Model")
    ax_bars.set_ylabel("Metric Value")
    #ax_bars.legend(title="Models")
    ax_bars.set_yscale('log')
    ax_bars.set_ylim(1e-2, 1e3)
    ax_bars.axhline(1)
    ax_bars.axhline(3)

    # Histogram of z-scores
    bins = np.linspace(-5, 5, 60)
    for i, z in enumerate(ZscoreSets):
        if z is not None:
            ax_hist_z.hist(z, bins=bins, histtype='step', alpha=1, lw=2, label=model_names[i], density=True, color=colors[i % len(colors)])

    ax_hist_z.set_title("Z-score Distribution")
    ax_hist_z.set_xlabel("Z-score")
    ax_hist_z.set_ylabel("Density")
    rv = scipy.stats.norm(loc=0, scale=1)
    ax_hist_z.plot(bins, rv.pdf(bins), ls='dotted')
    #ax_hist_z.legend()

    # Histogram of Sigmas
    #bins1 = np.linspace(0.1, 5, 30)
    #bins2 = np.logspace(-6, -1, 30)
    #bins = np.concatenate((bins2[bins2 < bins1[0]], bins1))
    bins = np.logspace(-6, 2, 100)

    for i, sigma in enumerate(SigmasSets):
        if sigma is not None:
            ax_hist_sigma.hist(sigma, bins=bins, histtype='step', alpha=1, lw=2, label=names[i], density=False)
    ax_hist_sigma.set_title("Sigma Distribution")
    ax_hist_sigma.set_xlabel("Sigma")
    ax_hist_sigma.set_ylabel("Counts?")
    ax_hist_sigma.set_xscale('log')
    #ax_hist_sigma.legend()

    
    bins = np.linspace(0, 5, 100)
    for i, sigma in enumerate(SigmasSets):
        if sigma is not None:
            ax_hist_sigma2.hist(sigma, bins=bins, histtype='step', alpha=1, lw=2, label=names[i], density=False)
    ax_hist_sigma2.set_title("Sigma Distribution")
    ax_hist_sigma2.set_xlabel("Sigma")
    ax_hist_sigma2.set_ylabel("Counts?")
    #ax_hist_sigma2.set_xscale('log')
    #ax_hist_sigma2.legend()

    
    for i, z in enumerate(ZscoreSets):
        abs_z = np.abs(z)
        sorted_abs_z = np.sort(abs_z.flatten())
        #print(sorted_abs_z)
        #theoretical_cdf = norm.cdf(sorted_abs_z)
        theoretical_cdf = 2 * norm.cdf(sorted_abs_z) - 1

        ax_hist_zCDF.plot(sorted_abs_z, np.linspace(0, 1, len(sorted_abs_z)), label='Empirical CDF')
    ax_hist_zCDF.plot(sorted_abs_z, theoretical_cdf, label='Theoretical N(0,1) CDF', ls='dashed', c='black')
    #ax_hist_zCDF.legend()
    ax_hist_zCDF.set_xlabel("|Z|")
    ax_hist_zCDF.set_ylabel("CDF")
    ax_hist_zCDF.set_xlim([0, 10])


    #ax_hist_zCDF.title("Empirical vs Theoretical CDF of |Z|")


    plt.tight_layout()
    plt.show()

def print_and_plot_summary_stats(model_names, 
                                 model_data_statistics, model_data_periodogram, model_data_Y, 
                                 params,
                                 kfold=False):

    
    print('\n\n\n## Test Data Version - with K-Folds')
    SumStats_list = []    
    flattened_name_list = []
    for i, model_name in enumerate(model_names):
        X, X_period, Y = model_data_statistics[i], model_data_periodogram[i], model_data_Y[i] 
        #X, X_period, Y = model_valid_data_statistics[i], model_valid_data_periodogram[i], model_valid_data_Y[i] 
        if kfold:
            for run_id in all_runs_log[model_name].keys():
                for fold_id in all_runs_log[model_name][run_id].keys():
                    # Now that params is stored under run_id, not all items are ints.
                    if isinstance(fold_id, str):
                        continue
                        
                    #if (config_dict['batch_size']==1000) and (config_dict['lr']<1e-3):
                    model = all_runs_log[model_name][run_id][fold_id]['model']
                    flattened_name_list.append(str(config_dict))
                    SumStats_list.append(SummaryStats(model, X, X_period, Y))
        else:  
                for run_id, config_dict in enumerate(ParameterGrid(params)):
                    #if (config_dict['batch_size']==1000) and (config_dict['lr']<1e-3):
                    model = all_runs_log[model_name][run_id][fold_id]['model']
                    flattened_name_list.append(str(config_dict))
                    SumStats_list.append(SummaryStats(model, X, X_period, Y))


# Commented out - this was example/test code at module level
# print('\n\n\n## Test Data Version - with K-Folds')
# SumStats_list = []
# flattened_name_list = []
# for i, model_name in enumerate(model_names):
#     X, X_period, Y = model_test_data_statistics[i], model_test_data_periodogram[i], model_test_data_Y[i]
#     #X, X_period, Y = model_valid_data_statistics[i], model_valid_data_periodogram[i], model_valid_data_Y[i]
#
#
#
#     PlotSummaryStats(SumStats_list, flattened_name_list)
#
#     return


#######################################################################################################################################
#######################################################################################################################################





import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def _get_kfold_array(run_data, metric_key):
    """Extract a metric from all folds as a [n_folds, n_epochs] array."""
    arrays = []
    for k, v in run_data.items():
        if isinstance(k, int):
            arrays.append(v['log'][metric_key])
    return np.array(arrays)


def plot_2d_statistics(
    config_dict,
    x_param,
    y_param,
    log_metrics,
    fold_agg='median',
    epoch_select='best',
    lam=0.5,
    figsize=(6, 5),
):
    """Plot 2D heatmaps of training-log metrics aggregated across k-folds.

    Parameters
    ----------
    config_dict : dict
        all_runs_log[model_name] — mapping of {run_id: run_data}.
    x_param, y_param : str
        Hyperparameter names for the x and y axes.
    log_metrics : list of str
        Training log keys to plot (e.g. ['valid_MAE', 'valid_RMSE']).
    fold_agg : str
        How to aggregate across folds: 'median', 'mean', 'min', or 'max'.
    epoch_select : str
        'best' selects the epoch minimising mean_loss + lam*std_loss across
        folds; 'final' uses the last epoch.
    lam : float
        Stability penalty weight for best-epoch selection.
    figsize : tuple
        Figure size per plot.
    """
    agg_funcs = {
        'median': np.nanmedian,
        'mean': np.nanmean,
        'min': np.nanmin,
        'max': np.nanmax,
    }
    agg_fn = agg_funcs[fold_agg]

    x_vals = sorted(set(run["params"][x_param] for run in config_dict.values()))
    y_vals = sorted(set(run["params"][y_param] for run in config_dict.values()))
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}

    for metric_key in log_metrics:
        Z = np.full((len(y_vals), len(x_vals)), np.nan)

        for run in config_dict.values():
            metric_folds = _get_kfold_array(run, metric_key)  # [n_folds, n_epochs]

            if epoch_select == 'best':
                loss_folds = _get_kfold_array(run, 'valid_Loss')
                mean_loss = np.nanmean(loss_folds, axis=0)
                std_loss = np.nanstd(loss_folds, axis=0)
                epoch_idx = int(np.nanargmin(mean_loss + lam * std_loss))
            else:
                epoch_idx = -1

            fold_values = metric_folds[:, epoch_idx]
            agg_val = float(agg_fn(fold_values))

            x_i = x_map[run["params"][x_param]]
            y_i = y_map[run["params"][y_param]]
            Z[y_i, x_i] = agg_val

        plt.figure(figsize=figsize)

        positive = Z[Z > 0]
        if positive.size == 0:
            norm = None
        else:
            norm = mcolors.LogNorm(
                vmin=np.nanmin(positive),
                vmax=min(np.nanmax(Z), 1e20),
            )

        im = plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis', norm=norm)
        plt.colorbar(im, label=metric_key)

        x_labels = [f"{x:.3g}" if isinstance(x, float) else str(x) for x in x_vals]
        y_labels = [f"{y:.3g}" if isinstance(y, float) else str(y) for y in y_vals]
        plt.xticks(ticks=np.arange(len(x_vals)), labels=x_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(y_vals)), labels=y_labels)

        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                val = Z[yi, xi]
                if not np.isnan(val) and norm is not None:
                    text_color = "white" if norm(val) < 0.5 else "black"
                    plt.text(xi, yi, f"{val:.2g}", ha="center", va="center",
                             color=text_color, fontsize=8)

        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"{metric_key} vs {x_param} & {y_param}\n({fold_agg}, epoch={epoch_select})")
        plt.tight_layout()
        plt.show()


def plot_2d_statistics_wrapper(
    all_runs_log, x_param, y_param,
    log_metrics=None,
    fold_agg='median',
    epoch_select='best',
    lam=0.5,
    **kwargs,
):
    """Convenience wrapper that plots log-based 2D heatmaps for every model.

    Parameters
    ----------
    all_runs_log : dict
        Top-level log dict: {model_name: {run_id: run_data}}.
    log_metrics : list of str or None
        Metrics to plot. Defaults to a standard set of validation metrics.
    """
    if log_metrics is None:
        log_metrics = [
            'valid_MAE', 'valid_RMSE', 'valid_Loss',
            'valid_median_sigma', 'valid_Coverage68',
            'valid_Coverage95', 'valid_CRPS',
        ]
    for model_name, runs in all_runs_log.items():
        print(f"--- {model_name} ---")
        plot_2d_statistics(
            config_dict=runs,
            x_param=x_param,
            y_param=y_param,
            log_metrics=log_metrics,
            fold_agg=fold_agg,
            epoch_select=epoch_select,
            lam=lam,
            **kwargs,
        )



###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################




import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import itertools


def permutation_importance(model, X_val, y_val, metric, feature_names=None, n_repeats=5, plot=True, corr_threshold=0.8):
    with torch.no_grad():
        model.eval()
        
        y_pred, sigma_pred = model(X_val)
        baseline = metric(y_val, y_pred)
        importances = []

        for i in range(X_val.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_perm = X_val.clone()
                X_perm[:, i] = X_perm[torch.randperm(X_perm.size(0)), i]
                y_pred, sigma_pred = model(X_perm)
                score = metric(y_val, y_pred)
                scores.append(score)
            importances.append(np.mean(scores) - baseline)

        importances = np.array(importances)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        # Sort by importance descending
        sorted_indices = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        # Compute correlation matrix and group features
        X_np = X_val.detach().cpu().numpy()
        X_scaled = StandardScaler().fit_transform(X_np)
        corr_matrix = np.corrcoef(X_scaled, rowvar=False)

        # Group features by correlation threshold
        groups = []
        assigned = set()
        for i in range(len(corr_matrix)):
            if i in assigned:
                continue
            group = {i}
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > corr_threshold:
                    group.add(j)
            assigned |= group
            groups.append(group)

        # Assign a color to each group
        color_palette = sns.color_palette("tab10", n_colors=len(groups))
        feature_to_color = {}
        for color, group in zip(color_palette, groups):
            for idx in group:
                feature_to_color[idx] = color

        # Create a list of colors for the sorted features
        sorted_colors = [feature_to_color[i] for i in sorted_indices]

        if plot:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(sorted_names)), sorted_importances, color=sorted_colors)
            plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
            plt.ylabel("Increase in error (importance)")
            plt.title("Permutation Feature Importance")
            plt.tight_layout()
            plt.show()

    return list(zip(sorted_names, sorted_importances))


def permutation_feature_importance_wrapper(all_runs_log, model_names, params, model_stat_data, model_period_data, model_Y_data):
    """
    This function has not yet been updated to work with periodogram data/models. 
    Don't use until updated
    """
    print('\n\n\n## Test Data Version')
    SumStats_list = []    
    flattened_name_list = []
    for i, model_name in enumerate(model_names):
        #X_train, X_test, y_train, y_test, X_test_BDATA, y_test_BDATA, ATTRIBUTES, SCALER, POLY_SCALER = model_data[i]
        for run_id, config_dict in enumerate(ParameterGrid(params)):
            #if (config_dict['lr']==0.0001) and (config_dict['batch_size']==100):
            #if (config_dict['lr']==0.0001) and (config_dict['batch_size']==100) and (config_dict['decoupled']==False):
            #if (config_dict['Layer1_Size']==32) and (config_dict['decoupled']):
            if (config_dict['lr']==params['lr'][1]) and (config_dict['batch_size']==1000):
            #if (config_dict['batch_size']==1000):
    
            #if not (config_dict['decoupled']):
            #if True:
                model = all_runs_log[model_name][run_id]['model']
                flattened_name_list.append(str(config_dict))
                
                X = X_test_BDATA[::10, :]
                Y = y_test_BDATA[::10, :]#.numpy().flatten()
    
                permutation_importance(model, X, Y, rmse, feature_names=ATTRIBUTES, n_repeats=2, plot=True)
    
            #break

    return


#permutation_importance(model, X_test, y_test, rmse, feature_names=ATTRIBUTES, n_repeats=5, plot=True):

