import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from itertools import cycle




def normalize_to_kfold(all_runs_log):
    """
    Convert a single-run log into a k-fold-shaped structure
    so the rest of the code can treat everything uniformly.
    """

    # If it's already k-fold shaped (multiple models)
    if any(isinstance(v, dict) and "params" in v and any(isinstance(k, int) for k in v)
           for v in all_runs_log.values()):
        return all_runs_log  # nothing to change

    # Otherwise → wrap into expected structure:
    # Create: { "single_model": { 0: { "params": ..., 0: {"log": ...} } } }
    wrapped = {
        "single_model": {
            0: {  # run_id
                "params": all_runs_log["params"],
                0: {"log": all_runs_log["log"]}  # fold_id
            }
        }
    }
    return wrapped


def plot_trainval_loss_evolution(all_runs_log, all_models_same_plot=False):
    # Function outptus the Training Loss, TrainingLoss Components, and Validation Loss for each model run per epoch.
    # If all_models_same_plot=True, then all models and all kfolds are plotted on the same plot.
    # If all_models_same_plot=True, then each model on a seperate plot, but all kfolds still on same plot.
    
    # Case 1: k-fold results (model_name → runs)
    is_kfold = any(isinstance(v, dict) and "params" in v for v in all_runs_log.values())

    if is_kfold:
        # ============================================================
        # K-FOLD CASE
        # all_runs_log = { model_name: { run_id: run_data, ... }, ... }
        # ============================================================


        for model_name, runs in all_runs_log.items():
            print(model_name)

            if not all_models_same_plot:
                fig = plt.figure(figsize=(12, 10 + int(1 * len(runs))))
                fig.suptitle(f"Training Log for Model: {model_name}", fontsize=16)

                gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 1])
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                ax3 = fig.add_subplot(gs[2])
                ax_legend = fig.add_subplot(gs[3])

            colors = plt.cm.tab10.colors
            color_cycle = cycle(colors)
            legend_handles = []

            for run_id, run_data in runs.items():
                color = next(color_cycle)
                params = run_data["params"]
                label = str(params)

                # Detect fold keys
                fold_logs = [
                    fold_data["log"]
                    for k, fold_data in run_data.items()
                    if isinstance(k, int)
                ]

                for log in fold_logs:
                    ax1.plot(log["epoch"], log["mean_batch_loss"], label=label, color=color)
                    ax2.plot(log["epoch"], log["train_errLoss"], color=color, linestyle='solid')
                    ax2.plot(log["epoch"], np.asarray(log["train_sigmaLoss"]), color=color, linestyle="--")
                    ax3.plot(
                        log["epoch"],
                        np.asarray(log["valid_errLoss"]) + np.asarray(log["valid_sigmaLoss"]),
                        color=color, linestyle='solid'
                    )

                legend_handles.append(mlines.Line2D([], [], color=color, label=label))

    else:
        # ============================================================
        # SINGLE MODEL (NO K-FOLD)
        # all_runs_log = { 'params':..., 'model':..., 'log':... }
        # ============================================================

        print("single-model")

        fig = plt.figure(figsize=(12, 10))
        fig.suptitle("Training Log", fontsize=16)

        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax_legend = fig.add_subplot(gs[3])

        color = plt.cm.tab10.colors[0]
        legend_handles = []

        params = all_runs_log["params"]
        label = str(params)
        log = all_runs_log["log"]

        # Same unified plotting logic
        ax1.plot(log["epoch"], log["mean_batch_loss"], label=label, color=color)
        ax2.plot(log["epoch"], log["train_errLoss"], color=color, linestyle='solid')
        ax2.plot(log["epoch"], np.asarray(log["train_sigmaLoss"]), color=color, linestyle="--")
        ax3.plot(
            log["epoch"],
            np.asarray(log["valid_errLoss"]) + np.asarray(log["valid_sigmaLoss"]),
            color=color,
            linestyle='solid'
        )

        legend_handles.append(mlines.Line2D([], [], color=color, label=label))

    ax1.set_title("Training Loss - Total\nsolid=AvgBatch, dotted=Total")
    ax1.set_ylabel("Training Loss")
    ax1.set_yscale('symlog', linthresh=0.1, linscale=0.1)
    ax1.set_ylim(-10, 100)
    ax1.set_xlabel("Epoch")

    ax2.set_title("Train Loss - Components")
    ax2.set_ylabel("Train Loss")
    ax2.set_yscale('symlog', linthresh=0.1, linscale=0.1)
    ax2.set_ylim(-10, 10)
    ax2.set_xlabel("Epoch")

    ax3.set_title("Validation Loss - Total")
    ax3.set_ylabel("Validation Loss")
    ax3.set_xlabel("Epoch")
    #ax3.set_yscale('log')
    #ax3.set_ylim(0.01, 100)
    ax3.set_yscale('symlog', linthresh=0.1, linscale=0.1)
    ax3.set_ylim(-10, 10)

    #ax3.set_yscale('symlog')

    # Turn off axis and show legend in bottom axis
    ax_legend.axis("off")
    ax_legend.legend(handles=legend_handles, loc="center", ncol=1, fontsize='small', frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if not all_models_same_plot:
        plt.show()
    plt.show()
    
    return



# Now plot all the metrics we care about vs. epoch with statistical treatment of k-folds.
def get_high_contrast_colors(n):
    """Return n high-contrast RGB colors."""
    kelly_colors = np.array([
        (0.90, 0.17, 0.31),   # red
        (0.00, 0.60, 0.20),   # green
        (1.00, 0.70, 0.00),   # orange
        (0.00, 0.48, 1.00),   # blue
        (0.58, 0.00, 0.83),   # purple
        (0.80, 0.00, 0.80),   # magenta
        (0.00, 0.80, 0.80),   # cyan
        (0.40, 0.40, 0.40),   # gray
        (1.00, 0.90, 0.00),   # yellow
        (1.00, 0.40, 0.00),   # deep orange
        (0.20, 0.60, 0.80),   # steel blue
        (0.35, 0.20, 0.70),   # violet
        (0.60, 0.60, 0.20),   # olive
        (0.94, 0.50, 0.50),   # light red
        (0.60, 0.20, 0.20),   # brown
        (0.20, 0.20, 0.20),   # dark gray
        (0.80, 0.60, 0.70),   # pinkish
        (0.10, 0.40, 0.60),   # slate
        (0.40, 0.70, 0.20),   # light green
        (0.20, 0.50, 0.00),   # deep green
        (0.70, 0.70, 0.10),   # mustard
        (0.50, 0.70, 0.50),   # pale green
    ])

    if n <= len(kelly_colors):
        return kelly_colors[:n]

    # repeat if more needed
    reps = (n // len(kelly_colors)) + 1
    return np.vstack([kelly_colors] * reps)[:n]


def get_param_label(all_param_dicts, current_params):
    """
    Create a label string showing ONLY the parameters that vary across runs.
    `all_param_dicts` is a list of dicts (one for each run).
    `current_params` is the dict for the run being plotted.
    """

    # Determine which parameters vary across models
    varying_params = []
    all_keys = current_params.keys()

    for key in all_keys:
        # Collect values of this key from all runs
        values = {p[key] for p in all_param_dicts if key in p}
        if len(values) > 1:          # varies across models
            varying_params.append(key)

    # Construct label string using only varying params
    pieces = []
    for key in varying_params:
        pieces.append(f"{key}={current_params[key]}")
    return ", ".join(pieces)


def plot_all_metric_evolution(all_runs_log, 
                               specific_lr=None, specific_batchsize=None, specific_dropout=None,
                               use_minmax_spread=False,
                               k_fold=True
    ):
    ''' 
    This will plot all the models as the metrics listed below vs. epoch using the median value of all k-folds 
    and the std of the kfolds as error bars. 
    the specific_x inputs will make sure that only models with that specific value of that parameter are plotting,
    useful for visualizing a subselection of models when you have a large grid of models across parameter space.
    '''

    if not k_fold:
        all_runs_log = normalize_to_kfold(all_runs_log)
    
    list_of_metricsA = [
                       'train_Loss', 'mean_batch_loss',
                       'train_Loss', 'valid_Loss', 
                       'train_errLoss', 'valid_errLoss', 
                       'train_sigmaLoss', 'valid_sigmaLoss', 
                      ] 
    
    list_of_metricsB = [
                       'train_RMSE', 'valid_RMSE',
                       'train_MAE',  'valid_MAE',
                        'train_median_sigma', 'valid_median_sigma',
                        'train_mean_sigma', 'valid_mean_sigma',
                      ] 
    
    list_of_metricsC = [
                        'train_Coverage68', 'valid_Coverage68',
                        'train_Coverage95', 'valid_Coverage95',
                        'train_CRPS', 'valid_CRPS',
                      ] 


    for list_of_metrics in [list_of_metricsA, list_of_metricsB, list_of_metricsC]:
        # --- Collect all parameter dicts so we can detect which vary for the legend label ---
        all_param_dicts = []
        for model_name, runs in all_runs_log.items():
            for run_id, run_data in runs.items():
                if isinstance(run_id, str):
                    continue
                all_param_dicts.append(run_data["params"])

            
        num_metrics = len(list_of_metrics)
    
        # Define the plot
        fig, axs = plt.subplots(num_metrics//2, 2, figsize=(12, 6*(num_metrics//2)))
        axs = axs.flatten()
        for metric_id, metric in enumerate(list_of_metrics):
             axs[metric_id].set_title(metric)
             axs[metric_id].set_ylabel(metric, fontsize=16)
             axs[metric_id].set_xlabel('epoch')
             axs[metric_id].set_yscale('log')
    
        # Loop through list
        for model_name, runs in all_runs_log.items():
            print(model_name)
    
            np.random.seed(42)
            # colors = 0.5 + 0.5 * np.random.rand(len(runs), 3)   # pastel-ish
            colors = get_high_contrast_colors(len(runs))
            color_cycle = cycle(colors)
            legend_handles = []
            
            for run_id, run_data in runs.items():
                print('Run_id', run_id)
                params = run_data["params"]

                if (specific_lr!=None) and (params['lr']!=specific_lr):
                    continue
                if (specific_batchsize!=None) and (params['batch_size']!=specific_batchsize):
                    continue
                if (specific_dropout!=None) and (params['dropout_prob']!=specific_dropout):
                    continue

                color = next(color_cycle)
                label = get_param_label(all_param_dicts, params)

    
                n_epochs = params['n_epochs']
                n_kfolds = params['num_kfolds']
                
                for metric_id, metric in enumerate(list_of_metrics):
                    temporary_array  = np.empty((n_kfolds, n_epochs))
                    
                    for fold_id, fold_data in run_data.items():
                        if isinstance(fold_id, str):                     # Now that params is stored under run_id, not all items are ints.
                            continue
                        log = fold_data["log"]    
                        temporary_array[fold_id, :] = log[metric]
    
                    # Plot it
                    x = np.arange(0, n_epochs, 1); 
                    y = np.median(temporary_array, axis=0)
                    y_std = np.std(temporary_array, axis=0)
                    #mad = np.median(np.abs(temporary_array - y), axis=0)
                    #y_spread = 1.4826 * mad
                    #print('COMPARE MAD VS STD', y_std, y_spread)
                    
                    y_upper = y + y_std; y_lower = y - y_std
                    #y_upper = np.max(temporary_array, axis=0); y_lower = np.min(temporary_array, axis=0)
    
                    axs[metric_id].plot(x, y, color=color, label=label)

                    if use_minmax_spread:
                        # --- MIN–MAX spread as dashed lines ---
                        y_min_line = np.min(temporary_array, axis=0)
                        y_max_line = np.max(temporary_array, axis=0)

                        axs[metric_id].plot(x, y_min_line, color=color, linestyle='--', linewidth=1)
                        axs[metric_id].plot(x, y_max_line, color=color, linestyle='--', linewidth=1)
                    else:
                        # --- STD shaded region ---
                        y_upper = y + y_std
                        y_lower = y - y_std
                        axs[metric_id].fill_between(x, y_lower, y_upper, color=color, alpha=0.3)
                    
                    if np.any(y<=0): #change from log to linear scale if the metric goes negative
                        axs[metric_id].set_yscale('symlog')

                    y_min = np.min(y) - 1
                    y_max = np.max(y) * 2
                    
                    if np.isnan(y_min) or np.isnan(y_max):
                        print('NaNs detected in data, careful.')
                        y_min = np.nanmin(y) - 1
                        y_max = np.nanmax(y) * 2
                    
                    if np.isinf(y_min) or np.isinf(y_max):
                        print('Infs detected in data, careful.')
                        print('      Not setting y bounds.')
                    else:
                        current_ymin, current_ymax = axs[metric_id].get_ylim()

                        # Update only if expansion needed
                        if y_min < current_ymin:
                            new_ymin = y_min
                        else: 
                            new_ymin = current_ymin
                            
                        if y_max > current_ymax:
                            new_ymax = y_max
                        else:
                            new_ymax = current_ymax

                        if new_ymax > 1e5:
                            new_ymax =1e5
                        if new_ymin < -1e5:
                            new_ymin = -1e5

                        
                        
                        # Apply only if either bound changed
                        if new_ymin != current_ymin or new_ymax != current_ymax:
                            axs[metric_id].set_ylim(new_ymin, new_ymax)
    
                    #if 'loss' in metric:
                        
                    if 'Coverage95' in metric:
                        axs[metric_id].axhline(0.95)
                        best_metric = np.min(np.abs(np.median(temporary_array, axis=0)-0.95))
                        epoch_of_best_metric = np.argmin(np.abs(np.median(temporary_array, axis=0)-0.95))
                        axs[metric_id].set_ylim([0.85, 1])
                        axs[metric_id].set_yscale('linear')
                        axs[metric_id].annotate( 'under confident', xy=(0.6, 0.9), xycoords='axes fraction', ha='center', va='center' )
                        axs[metric_id].annotate( 'over confident', xy=(0.6, 0.1), xycoords='axes fraction', ha='center', va='center')                
                    elif 'Coverage68' in metric:
                        axs[metric_id].axhline(0.68)
                        best_metric = np.min(np.abs(np.median(temporary_array, axis=0)-0.68))
                        epoch_of_best_metric = np.argmin(np.abs(np.median(temporary_array, axis=0)-0.68))
                        axs[metric_id].set_ylim([0.5, 1])
                        axs[metric_id].set_yscale('linear')
                        axs[metric_id].annotate( 'under confident', xy=(0.6, 0.9), xycoords='axes fraction', ha='center', va='center' )
                        axs[metric_id].annotate( 'over confident', xy=(0.6, 0.1), xycoords='axes fraction', ha='center', va='center')                
                    elif 'CRPS' in metric:
                        axs[metric_id].axhline(0)
                        best_metric = np.min(np.abs(np.median(temporary_array, axis=0)))
                        epoch_of_best_metric = np.argmin(np.abs(np.median(temporary_array, axis=0)))
                        axs[metric_id].set_ylim([0.05, 1])
                        axs[metric_id].annotate( 'lower is better', xy=(0.6, 0.1), xycoords='axes fraction', ha='center', va='center')                
                    else:
                        best_metric = np.min(np.median(temporary_array, axis=0))
                        epoch_of_best_metric = np.argmin(np.median(temporary_array, axis=0))
                    
                    print(f'      {metric} = {best_metric:2f}; epoch = {epoch_of_best_metric}')
    
        plt.tight_layout()

        # --- Make a single combined legend at bottom ---
        handles_labels = {}
        for ax in axs:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles_labels[l] = h  # deduplicate

        if len(handles_labels) > 0:
            fig.legend(handles_labels.values(),
                    handles_labels.keys(),
                    loc='lower center',
                    bbox_to_anchor=(0.5, 0.05),
                    ncol=3, fontsize=12, handleheight=4)
            plt.subplots_adjust(bottom=0.15)

        plt.show()
    
    return
    