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

import numpy as np
import torch
import pickle
import copy
import sys
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import logging

# Emergency crash log
import atexit
def emergency_exit_log():
    with open(project_dir_path + 'TempFiles/logs/EMERGENCY_EXIT.log', 'a') as f:
        import traceback
        f.write(f"\n{'='*70}\n")
        f.write(f"Process exiting at: {datetime.now()}\n")
        f.write(f"Stack trace:\n")
        traceback.print_stack(file=f)
        f.write(f"{'='*70}\n")
        f.flush()
atexit.register(emergency_exit_log)

# Threading for heartbeat
import threading

# Global variables for heartbeat control
_heartbeat_stop_event = threading.Event()
_heartbeat_thread = None

def write_pid_file():
    """Write current process PID to a file for monitoring."""
    pid_file = project_dir_path + 'TempFiles/logs/training.pid'
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(f"{os.getpid()}\n")
        f.write(f"Started: {datetime.now()}\n")
    return pid_file

def heartbeat_writer(interval=30):
    """
    Write a heartbeat timestamp to a file every `interval` seconds.
    This runs in a daemon thread so it won't prevent process exit.
    """
    heartbeat_file = project_dir_path + 'TempFiles/logs/heartbeat.txt'
    os.makedirs(os.path.dirname(heartbeat_file), exist_ok=True)

    count = 0
    while not _heartbeat_stop_event.is_set():
        try:
            with open(heartbeat_file, 'w') as f:
                f.write(f"Heartbeat #{count}\n")
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Status: ALIVE\n")
                f.flush()
            count += 1
        except Exception as e:
            pass  # Silently ignore errors (file system issues, etc.)
        _heartbeat_stop_event.wait(interval)

def start_heartbeat():
    """Start the heartbeat writer thread."""
    global _heartbeat_thread
    _heartbeat_stop_event.clear()
    _heartbeat_thread = threading.Thread(target=heartbeat_writer, args=(30,), daemon=True)
    _heartbeat_thread.start()
    return _heartbeat_thread

def stop_heartbeat():
    """Stop the heartbeat writer thread."""
    _heartbeat_stop_event.set()
    if _heartbeat_thread:
        _heartbeat_thread.join(timeout=5)

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


def initialize_device(use_gpu):
    """Initialize device with error handling.

    Args:
        use_gpu: Boolean flag indicating whether to use GPU

    Returns:
        torch.device object (either 'cuda:0' or 'cpu')

    Raises:
        RuntimeError: If GPU requested but CUDA not available
    """
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU training requested but CUDA not available!\n"
                "Options:\n"
                "  1. Set use_gpu=False\n"
                "  2. Install CUDA-enabled PyTorch\n"
                "  3. Check GPU drivers"
            )
        device = torch.device('cuda:0')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")

        # Configure PyTorch to use all available CPU cores
        torch.set_num_threads(8)  # Use all 8 logical cores
        torch.set_num_interop_threads(8)  # Parallelism for operations
        os.environ['OMP_NUM_THREADS'] = '8'  # OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '8'  # MKL (Intel Math Kernel Library) threads
        logging.info(f"PyTorch threads: {torch.get_num_threads()}")
        logging.info(f"PyTorch interop threads: {torch.get_num_interop_threads()}")

    return device


def setup_logging(log_dir):
    """Setup logging to both console and file directly.

    File logging is independent of tee, ensuring logs are captured even if
    SSH disconnects or tee fails.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped log file for direct file logging
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Stream handler (stdout) - also captured by tee if running
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # File handler - writes directly to file, independent of tee
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Same format for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Force flush after every write
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[stream_handler, file_handler],
        force=True  # Reset any existing handlers
    )

    # Disable buffering on stdout and stderr (force immediate writes to tee)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)  # Also unbuffer stderr for exceptions

    return log_file, timestamp


def load_data(temp_files_path, dataset_name='traintest_data5.pkl'):
    """Load preprocessed training data

    Args:
        temp_files_path: Path to TempFiles directory
        dataset_name: Name of the dataset .pkl file (default: 'traintest_data5.pkl')
    """
    logging.info("Loading training data...")
    logging.info(f"Dataset: {dataset_name}")

    with open(temp_files_path + dataset_name, 'rb') as f:
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

    # Initialize device (extract use_gpu from params)
    # Since params values are lists for ParameterGrid, extract first value
    if params_config is None:
        print("NO CONFIG PROVIDED")
        return None

    use_gpu = params_config.get('use_gpu', [False])[0]
    device = initialize_device(use_gpu)

    # Extract dataset_name from config (not a list, just a string value)
    dataset_name = params_config.get('dataset_name', 'traintest_data5.pkl')

    # Load data
    (X_train, X_test, period_train, period_test,
     y_train, y_test, feature_cols,
     train_cluster_names, test_cluster_names) = load_data(temp_files_path, dataset_name)

    # Use provided params config, but remove non-grid params
    params = {k: v for k, v in params_config.items() if k not in ('dataset_name', 'use_gpu')}

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
            device=device,
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
    logging.info(f"Logs saved to: {log_file}")
    logging.info("="*70)

    return all_runs_log, output_path


if __name__ == "__main__":
    import traceback
    import signal

    # Signal handler to catch external termination
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        timestamp = datetime.now().isoformat()

        # Write to signal log file for post-mortem analysis
        signal_log_file = project_dir_path + 'TempFiles/logs/SIGNAL_RECEIVED.log'
        try:
            os.makedirs(os.path.dirname(signal_log_file), exist_ok=True)
            with open(signal_log_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"SIGNAL RECEIVED AT: {timestamp}\n")
                f.write(f"Signal number: {signum}\n")
                f.write(f"Signal name: {signal_name}\n")
                f.write(f"PID: {os.getpid()}\n")
                if signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGPIPE):
                    f.write(f"ACTION: IGNORED (continuing execution)\n")
                else:
                    f.write(f"ACTION: TERMINATING\n")
                f.write(f"{'='*70}\n")
                f.flush()
        except Exception:
            pass  # Don't let logging errors prevent signal handling

        # For SIGHUP, SIGTERM, and SIGPIPE, log but DON'T exit - we want to survive SSH disconnect
        if signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGPIPE):
            print(f"\n[{timestamp}] {signal_name} received - IGNORING, continuing training...",
                  file=sys.stderr, flush=True)
            return  # Continue running!

        # For other signals (SIGINT), exit as before
        print(f"\n{'='*70}", file=sys.stderr, flush=True)
        print(f"SIGNAL {signum} ({signal_name}) RECEIVED - PROCESS BEING TERMINATED", file=sys.stderr, flush=True)
        print(f"Time: {timestamp}", file=sys.stderr, flush=True)
        print(f"Signal logged to: TempFiles/logs/SIGNAL_RECEIVED.log", file=sys.stderr, flush=True)
        print(f"{'='*70}", file=sys.stderr, flush=True)

        # Stop heartbeat before exiting
        stop_heartbeat()

        sys.exit(128 + signum)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGHUP, signal_handler)   # SSH disconnect
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # Ignore broken pipe - CRITICAL for SSH disconnect

    # Check if config file was provided (for remote execution)
    config_file = project_dir_path + 'TempFiles/remote_config.pkl'

    try:
        print("="*70, flush=True)
        print("STARTING HYPERPARAMETER TRAINING", flush=True)
        print(f"PID: {os.getpid()}", flush=True)
        print("="*70, flush=True)

        # Write PID file and start heartbeat for monitoring
        pid_file = write_pid_file()
        print(f"PID file written to: {pid_file}", flush=True)

        start_heartbeat()
        print("Heartbeat thread started (30s interval)", flush=True)
        print("="*70, flush=True)

        if os.path.exists(config_file):
            # Don't log before setup_logging() is called - just load the config
            with open(config_file, 'rb') as f:
                params_config = pickle.load(f)
            run_hyperparameter_search(params_config)
        else:
            # Use default config
            run_hyperparameter_search()

        # Stop heartbeat and clean up
        stop_heartbeat()

        print("\n" + "="*70, flush=True)
        print("TRAINING COMPLETED SUCCESSFULLY", flush=True)
        print("="*70, flush=True)
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n" + "="*70, file=sys.stderr, flush=True)
        print("⚠️  TRAINING INTERRUPTED BY USER (Ctrl+C)", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        sys.exit(130)

    except MemoryError as e:
        print("\n" + "="*70, file=sys.stderr, flush=True)
        print("💥 FATAL ERROR: OUT OF MEMORY", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        print(f"Error: {e}", file=sys.stderr, flush=True)
        print("\nTroubleshooting:", file=sys.stderr, flush=True)
        print("  1. Reduce batch_size in parameter grid", file=sys.stderr, flush=True)
        print("  2. Reduce num_kfolds", file=sys.stderr, flush=True)
        print("  3. Use smaller model (fewer layers/units)", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(137)

    except RuntimeError as e:
        print("\n" + "="*70, file=sys.stderr, flush=True)
        print("💥 FATAL RUNTIME ERROR", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        print(f"Error: {e}", file=sys.stderr, flush=True)

        # Check for GPU/CUDA errors
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print("\n⚠️  GPU OUT OF MEMORY", file=sys.stderr, flush=True)
            print("Troubleshooting:", file=sys.stderr, flush=True)
            print("  1. Reduce batch_size", file=sys.stderr, flush=True)
            print("  2. Use CPU instead (set use_gpu=False)", file=sys.stderr, flush=True)

        print("="*70, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*70, file=sys.stderr, flush=True)
        print("💥 FATAL UNEXPECTED ERROR", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        print(f"Error type: {type(e).__name__}", file=sys.stderr, flush=True)
        print(f"Error message: {e}", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        print("\nFull traceback:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print("="*70, file=sys.stderr, flush=True)
        sys.exit(1)
