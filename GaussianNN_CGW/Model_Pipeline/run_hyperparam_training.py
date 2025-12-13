#!/usr/bin/env python
"""
Standalone script for hyperparameter training that can be run locally or remotely.

This script loads data, runs a grid search over hyperparameters using K-fold cross-validation,
and saves the results with timestamps.
"""

# IMPORTANT: Set matplotlib backend BEFORE any imports that use matplotlib
# This prevents hanging on headless systems (SSH/WSL without display)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to files, doesn't show)

from urllib.parse import non_hierarchical
import numpy as np
import torch
import pickle
import copy
import sys
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import logging

# Configure PyTorch to use all available CPU cores
# Set before any torch operations
torch.set_num_threads(8)  # Use all 8 logical cores
torch.set_num_interop_threads(8)  # Parallelism for operations
os.environ['OMP_NUM_THREADS'] = '8'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '8'  # MKL (Intel Math Kernel Library) threads

# Add project directory to path
project_dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(project_dir_path)

# Import custom modules
from GaussianNN_wPeriodogram import DualInputNN
from Training_HelperFcns import (
    GaussianNLLLoss_ME,
    sanitize_batch,
    Run_Single_Batch,
    Run_Single_Epoch,
    Train_Model,
    log_model_progress,
    plot_run_log
)

# Import K-fold training functions from shared module
from Kfold_HelperFcns import kfold_cross_validation, verify_kfold_split


def setup_logging(log_dir):
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file, timestamp


def load_data(temp_files_path):
    """Load preprocessed training data"""
    logging.info("Loading training data...")

    file_name = 'traintest_data4.pkl'
    with open(temp_files_path + file_name, 'rb') as f:
        PROCESSED_DATASET = pickle.load(f)

    (X_train, X_test,
     period_train, period_test,
     y_train, y_test,
     feature_cols,
     train_cluster_names, test_cluster_names) = PROCESSED_DATASET

    logging.info(f"Data loaded - X_train: {X_train.shape}, X_test: {X_test.shape}")
    logging.info(f"Features: {feature_cols}")

    return (X_train, X_test, period_train, period_test,
            y_train, y_test, feature_cols, train_cluster_names, test_cluster_names)


def run_hyperparameter_search(params_config=None):
    """
    Main function to run hyperparameter grid search

    Args:
        params_config: Optional dict with parameter grid. If None, uses default.
    """
    # Setup paths
    temp_files_path = project_dir_path + 'TempFiles/'
    log_dir = temp_files_path + 'logs/'

    # Setup logging
    log_file, timestamp = setup_logging(log_dir)
    logging.info("="*70)
    logging.info("HYPERPARAMETER TRAINING STARTED")
    logging.info("="*70)
    logging.info(f"Log file: {log_file}")
    logging.info(f"PyTorch threads: {torch.get_num_threads()}")
    logging.info(f"PyTorch interop threads: {torch.get_num_interop_threads()}")
    logging.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    logging.info(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")

    # Load data
    (X_train, X_test, period_train, period_test,
     y_train, y_test, feature_cols,
     train_cluster_names, test_cluster_names) = load_data(temp_files_path)

    # Define hyperparameter grid (use provided config or default)
    if params_config is None:
       print("NO CONFIG PROVIDED")
       return None
    else:
        params = params_config

    logging.info(f"Parameter grid: {params}")
    logging.info(f"Total configurations: {len(list(ParameterGrid(params)))}")

    # Initialize logging structure
    empty_train_log = {
        "epoch": [],
        "mean_batch_loss": [],
        "train_errLoss": [],
        "train_sigmaLoss": [],
        "train_Loss": [],
        "train_MAE": [],
        "train_RMSE": [],
        "train_median_sigma": [],
        "train_mean_sigma": [],
        "train_Coverage68": [],
        "train_Coverage95": [],
        "train_CRPS": [],
        "valid_errLoss": [],
        "valid_sigmaLoss": [],
        "valid_Loss": [],
        "valid_MAE": [],
        "valid_RMSE": [],
        "valid_median_sigma": [],
        "valid_mean_sigma": [],
        "valid_Coverage68": [],
        "valid_Coverage95": [],
        "valid_CRPS": [],
    }

    model_name = 'HyperparamSearch'
    all_runs_log = {
        model_name: {
            run_id: {
                **{"params": config_dict},
                **{
                    k_id: {
                        "model": None,
                        "log": copy.deepcopy(empty_train_log)
                    } for k_id in range(config_dict['num_kfolds'])
                }
            } for run_id, config_dict in enumerate(ParameterGrid(params))
        }
    }

    # Run training for each hyperparameter configuration
    run_id = 0
    total_runs = len(list(ParameterGrid(params)))
    grid_search_start = datetime.now()

    for Params in ParameterGrid(params):
        run_start = datetime.now()
        logging.info("\n" + "="*70)
        logging.info(f"[{run_start.strftime('%H:%M:%S')}] RUN #{run_id+1}/{total_runs} STARTING")
        logging.info(f"Params: {Params}")
        logging.info("="*70)

        # Run K-fold cross-validation
        all_runs_log[model_name][run_id] = kfold_cross_validation(
            X_train, period_train, train_cluster_names, y_train,
            DualInputNN, Params,
            all_runs_log[model_name][run_id],
            seed=42
        )

        # Plot run log (save to file for remote execution)
        try:
            # Create runtime_figs directory for saving plots
            figs_dir = temp_files_path + 'runtime_figs/'
            os.makedirs(figs_dir, exist_ok=True)

            # Generate filename based on run parameters
            fig_filename = f"run_{run_id}_lr{Params['lr']}_bs{Params['batch_size']}.png"
            fig_path = os.path.join(figs_dir, fig_filename)

            # Save plot to file instead of displaying
            plot_run_log(all_runs_log[model_name][run_id], save_path=fig_path)
            logging.info(f"   Plot saved to: {fig_filename}")
        except Exception as e:
            logging.warning(f"Could not generate plots: {e}")

        run_end = datetime.now()
        run_time = (run_end - run_start).total_seconds()
        total_elapsed = (run_end - grid_search_start).total_seconds()
        avg_run_time = total_elapsed / (run_id + 1)
        remaining_time = avg_run_time * (total_runs - run_id - 1)

        logging.info("="*70)
        logging.info(f"[{run_end.strftime('%H:%M:%S')}] RUN #{run_id+1}/{total_runs} COMPLETE")
        logging.info(f"   Run time: {run_time/60:.1f}min, Avg: {avg_run_time/60:.1f}min")
        logging.info(f"   Total elapsed: {total_elapsed/60:.1f}min, Est. remaining: {remaining_time/60:.1f}min")
        logging.info("="*70 + "\n")

        run_id += 1

    # Save results with timestamp
    output_dir = temp_files_path + 'ModelGridLogs/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'training_results_{timestamp}.pkl'
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, 'wb') as f:
        pickle.dump(all_runs_log, f)

    logging.info("="*70)
    logging.info("TRAINING COMPLETE!")
    logging.info(f"Results saved to: {output_path}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*70)

    return all_runs_log, output_path


if __name__ == "__main__":
    # Check if config file was provided (for remote execution)
    config_file = project_dir_path + 'TempFiles/remote_config.pkl'

    if os.path.exists(config_file):
        logging.info(f"Loading parameter config from {config_file}")
        with open(config_file, 'rb') as f:
            params_config = pickle.load(f)
        run_hyperparameter_search(params_config)
    else:
        # Use default config
        run_hyperparameter_search()
