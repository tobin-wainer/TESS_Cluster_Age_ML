"""
K-Fold Cross-Validation Helper Functions

This module contains functions for running K-fold cross-validation with proper
per-fold scaling to prevent data leakage.

Shared by:
- 1_HyperparamTuning.ipynb (interactive notebook)
- run_hyperparam_training.py (standalone remote training script)
"""

import numpy as np
import torch
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import GroupKFold
from Training_HelperFcns import apply_preprocessing, Train_Model

# Import pipeline configuration for device defaults
try:
    from pipeline_config import config
    DEFAULT_DEVICE = config.device
except ImportError:
    DEFAULT_DEVICE = torch.device('cpu')


def kfold_cross_validation(
    X1, X2, X_name, y, model_class, Params, single_run_log, device=None, k=5, seed=42
):
    """
    K-fold cross-validation with proper per-fold scaling.

    Args:
        X1: Summary statistics tensor (RAW, unscaled)
        X2: Periodogram tensor (RAW, unscaled)
        X_name: Cluster names array for grouping
        y: Target values tensor (RAW, unscaled)
        model_class: Neural network class to instantiate
        Params: Dictionary of hyperparameters
        single_run_log: Dictionary structure to store training logs
        k: Number of folds (overridden by Params['num_kfolds'])
        seed: Random seed for reproducibility

    Returns:
        single_run_log: Updated with trained models and logs for each fold

    KEY FEATURE: X1, X2, y are RAW (unscaled) data.
    Scaling is applied inside each fold to prevent data leakage.
    """
    k = Params['num_kfolds']

    # Initialize device (use config default if not provided)
    if device is None:
        device = DEFAULT_DEVICE

    kfold_start = time.time()
    wall_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{wall_time}] Starting {k}-fold cross-validation on device: {device}")

    # Infer dimensions from data
    stats_dim = X1.shape[1]
    period_dim = X2.shape[1] if X2 is not None else 0

    X_name = np.asarray(X_name)  # Make groups a numpy array
    n_samples = X1.shape[0]

    assert X2.shape[0] == n_samples, f"Periodogram samples ({X2.shape[0]}) != Stats samples ({n_samples})"
    assert len(X_name) == n_samples, f"Cluster names ({len(X_name)}) != Stats samples ({n_samples})"

    # Setup GroupKFold - keeps clusters together
    unique_clusters = np.unique(X_name)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_clusters)

    cluster_to_order = {c: i for i, c in enumerate(unique_clusters)}
    groups = np.array([cluster_to_order[c] for c in X_name])

    gkf = GroupKFold(n_splits=k)
    fold_metrics = []

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(X1, y, groups=groups)):
        fold_start = time.time()
        wall_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*70}")
        print(f"[{wall_time}] FOLD {fold_id+1}/{k} STARTING")
        print(f"{'='*70}")

        # Extract raw data for this fold
        X1_train_raw = X1[train_idx]
        X1_val_raw = X1[val_idx]
        X2_train_raw = X2[train_idx] if X2 is not None else None
        X2_val_raw = X2[val_idx] if X2 is not None else None
        y_train_raw = y[train_idx]
        y_val_raw = y[val_idx]

        # Apply preprocessing with per-fold scaling (prevents data leakage!)
        (X1_train, X1_val,
         X2_train, X2_val,
         y_train, y_val,
         scaler, y_mean, mean_peak_strength) = apply_preprocessing(
            X1_train_raw, X1_val_raw,
            X2_train_raw, X2_val_raw,
            y_train_raw, y_val_raw,
            device=device
        )

        print(f"Validation clusters: {len(np.unique(X_name[val_idx]))} clusters")

        # Prepare data for training
        training_data = (X1_train, X2_train, y_train)
        validation_data = (X1_val, X2_val, y_val)

        # Extract cluster names for training fold (for gradient debug output)
        train_cluster_names_fold = [X_name[i] for i in train_idx]

        # Initialize model for this fold
        model_inputs = (
            stats_dim,
            period_dim,
            Params['Layer1_Size'],
            Params['dropout_prob'],
            Params['use_periodogram'],
            Params['use_cnn'],
            Params['learn_sigma']
        )

        model = model_class(*model_inputs)
        model = model.to(device)

        # Train the model
        training_log = single_run_log[fold_id]['log']
        model, training_log, _ = Train_Model(
            model, training_data, validation_data, Params, training_log,
            device=device, cluster_names=train_cluster_names_fold
        )

        # Save results
        single_run_log[fold_id]['model'] = model
        single_run_log[fold_id]['scaler'] = scaler
        single_run_log[fold_id]['y_mean'] = y_mean

        fold_time = time.time() - fold_start
        total_elapsed = time.time() - kfold_start
        avg_fold_time = total_elapsed / (fold_id + 1)
        remaining_time = avg_fold_time * (k - fold_id - 1)
        wall_time = datetime.now().strftime("%H:%M:%S")

        final_loss = training_log['mean_batch_loss'][-1] if training_log['mean_batch_loss'] else float('nan')
        train_loss = training_log['train_Loss'][-1] if training_log['train_Loss'] else float('nan')
        val_loss = training_log['valid_Loss'][-1] if training_log['valid_Loss'] else float('nan')

        print(f"\n{'='*70}")
        print(f"[{wall_time}] FOLD {fold_id+1}/{k} COMPLETE")
        print(f"   Final batch loss: {final_loss:.4f}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        print(f"   Fold time: {fold_time/60:.1f}min, Avg fold time: {avg_fold_time/60:.1f}min")
        print(f"   Total elapsed: {total_elapsed/60:.1f}min, Est. remaining: {remaining_time/60:.1f}min")
        print(f"{'='*70}\n")

    total_time = time.time() - kfold_start
    wall_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{wall_time}] K-FOLD CROSS-VALIDATION COMPLETE - Total time: {total_time/60:.1f}min")

    return single_run_log


def verify_kfold_split(X_name, k=5, seed=42):
    """
    Verify that K-fold splitting keeps clusters together and doesn't leak data.

    Tests:
    1. No clusters appear in both train and validation
    2. All data points are accounted for in each fold
    3. Each cluster's data points stay together (not split across train/val)

    Args:
        X_name: Array of cluster names
        k: Number of folds
        seed: Random seed

    Returns:
        bool: True if all tests pass, False otherwise
    """
    X_name = np.asarray(X_name)
    n_samples = len(X_name)

    # Shuffle clusters (same logic as kfold_cross_validation)
    unique_clusters = np.unique(X_name)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_clusters)

    cluster_to_order = {c: i for i, c in enumerate(unique_clusters)}
    groups = np.array([cluster_to_order[c] for c in X_name])

    gkf = GroupKFold(n_splits=k)

    print("="*70)
    print("K-FOLD SPLIT VERIFICATION")
    print("="*70)
    print(f"Total samples: {n_samples}")
    print(f"Total clusters: {len(unique_clusters)}")
    print(f"Number of folds: {k}")
    print()

    all_val_indices = set()

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(np.arange(n_samples), groups=groups)):
        print(f"--- Fold {fold_id + 1} ---")

        # Get cluster names in train and validation
        train_clusters = set(X_name[train_idx])
        val_clusters = set(X_name[val_idx])

        # Test 1: No overlap between train and val clusters
        overlap = train_clusters & val_clusters
        if len(overlap) > 0:
            print(f"  ❌ FAILED: {len(overlap)} clusters appear in BOTH train and val!")
            print(f"     Overlapping clusters: {overlap}")
            return False
        else:
            print(f"  ✅ PASS: No cluster overlap (train={len(train_clusters)}, val={len(val_clusters)})")

        # Test 2: All data points accounted for
        total_samples = len(train_idx) + len(val_idx)
        if total_samples != n_samples:
            print(f"  ❌ FAILED: Missing samples (train={len(train_idx)}, val={len(val_idx)}, total={total_samples}, expected={n_samples})")
            return False
        else:
            print(f"  ✅ PASS: All samples accounted for (train={len(train_idx)}, val={len(val_idx)})")

        # Test 3: Verify each cluster's data points stay together
        clusters_split_across_folds = []
        for cluster in unique_clusters:
            cluster_mask = X_name == cluster
            cluster_indices = np.where(cluster_mask)[0]

            in_train = set(cluster_indices) & set(train_idx)
            in_val = set(cluster_indices) & set(val_idx)

            if len(in_train) > 0 and len(in_val) > 0:
                clusters_split_across_folds.append(cluster)

        if len(clusters_split_across_folds) > 0:
            print(f"  ❌ FAILED: {len(clusters_split_across_folds)} clusters split across train/val!")
            print(f"     Split clusters: {clusters_split_across_folds[:5]}...")
            return False
        else:
            print(f"  ✅ PASS: All clusters stay together (not split)")

        # Track validation indices across folds
        all_val_indices.update(val_idx)
        print()

    # Final check: every sample appears in validation exactly once
    if len(all_val_indices) != n_samples:
        print(f"❌ OVERALL FAILED: Not all samples appear in validation (got {len(all_val_indices)}, expected {n_samples})")
        return False
    else:
        print(f"✅ OVERALL PASS: Every sample appears in validation exactly once across all folds")

    print("="*70)
    print("✅ ALL TESTS PASSED - K-fold split is correct!")
    print("="*70)
    return True
