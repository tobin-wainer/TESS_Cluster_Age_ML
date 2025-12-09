"""
Verification Script for Data Leakage Fix

This script verifies that:
1. The preprocessing function works correctly
2. Scalers are different for each k-fold (proving no data leakage)
3. Data loaded from pickle is unscaled (raw)
4. Validation data statistics are independent of training data

Run this script after regenerating the data with 0_ProcessDataset.ipynb
"""

import numpy as np
import torch
import pickle
import sys
from pathlib import Path

# Setup paths
project_dir_path = "/Users/howard_willard/Desktop/TESS_Cluster_Age_ML/GaussianNN_CGW/Model_Pipeline/"
temp_files_path = project_dir_path + 'TempFiles/'
sys.path.append(project_dir_path)

from Training_HelperFcns import apply_preprocessing


def check_color(passed):
    """Return check mark or X based on test result"""
    return "✅" if passed else "❌"


def test_1_data_is_unscaled():
    """Test 1: Verify loaded data is unscaled (raw)"""
    print("\n" + "="*70)
    print("TEST 1: Verify Data is Unscaled (Raw)")
    print("="*70)

    # Load data
    with open(temp_files_path + 'traintest_data4.pkl', 'rb') as f:
        data = pickle.load(f)

    # Unpack (should NOT include SCALER and y_mean anymore)
    try:
        X_train, X_test, period_train, period_test, y_train, y_test, feature_cols, train_names, test_names = data
        print(f"{check_color(True)} Data unpacking successful (9 items, no SCALER/y_mean)")
    except ValueError as e:
        print(f"{check_color(False)} Data unpacking failed: {e}")
        return False

    # Check that data appears unscaled (mean ~= 0, std ~= 1 would indicate scaling)
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0)

    # Raw data should NOT have mean near 0 and std near 1 for all features
    is_scaled = (torch.abs(X_mean) < 0.1).all() and (torch.abs(X_std - 1.0) < 0.1).all()

    print(f"\nData Statistics (first 3 features):")
    print(f"  Mean: {X_mean[:3].numpy()}")
    print(f"  Std:  {X_std[:3].numpy()}")
    print(f"{check_color(not is_scaled)} Data appears UNSCALED (not standardized)")

    # Check y is not de-meaned
    y_mean_val = y_train.mean().item()
    is_demeaned = abs(y_mean_val) < 0.1

    print(f"\nTarget Statistics:")
    print(f"  Mean: {y_mean_val:.4f}")
    print(f"{check_color(not is_demeaned)} Target appears NOT de-meaned (mean != 0)")

    return not is_scaled and not is_demeaned


def test_2_preprocessing_function():
    """Test 2: Verify apply_preprocessing() works correctly"""
    print("\n" + "="*70)
    print("TEST 2: Verify apply_preprocessing() Function")
    print("="*70)

    # Load raw data
    with open(temp_files_path + 'traintest_data4.pkl', 'rb') as f:
        X_train, X_test, period_train, period_test, y_train, y_test, feature_cols, _, _ = pickle.load(f)

    # Apply preprocessing
    try:
        (X_train_scaled, X_test_scaled,
         period_train_norm, period_test_norm,
         y_train_dm, y_test_dm,
         scaler, y_mean, mean_peak_strength) = apply_preprocessing(
            X_train, X_test,
            period_train, period_test,
            y_train, y_test
        )
        print(f"{check_color(True)} apply_preprocessing() executed successfully")
    except Exception as e:
        print(f"{check_color(False)} apply_preprocessing() failed: {e}")
        return False

    # Verify scaling worked
    X_scaled_mean = X_train_scaled.mean(dim=0)
    X_scaled_std = X_train_scaled.std(dim=0)

    is_properly_scaled = (torch.abs(X_scaled_mean) < 0.1).all() and (torch.abs(X_scaled_std - 1.0) < 0.1).all()

    print(f"\nScaled Data Statistics (first 3 features):")
    print(f"  Mean: {X_scaled_mean[:3].numpy()}")
    print(f"  Std:  {X_scaled_std[:3].numpy()}")
    print(f"{check_color(is_properly_scaled)} Scaled data has mean~0, std~1")

    # Verify de-meaning worked
    y_demeaned_mean = y_train_dm.mean().item()
    is_properly_demeaned = abs(y_demeaned_mean) < 0.01

    print(f"\nDe-meaned Target Statistics:")
    print(f"  Mean: {y_demeaned_mean:.6f}")
    print(f"  y_mean value: {y_mean:.4f}")
    print(f"{check_color(is_properly_demeaned)} De-meaned target has mean~0")

    # Verify periodogram normalization
    if period_train_norm is not None:
        max_val = period_train_norm.max().item()
        print(f"\nPeriodogram Statistics:")
        print(f"  Max value: {max_val:.4f}")
        print(f"  Normalization factor: {mean_peak_strength:.6f}")
        print(f"{check_color(max_val < 10)} Periodogram appears normalized")

    return is_properly_scaled and is_properly_demeaned


def test_3_kfold_scalers_differ():
    """Test 3: Verify different k-folds produce different scalers"""
    print("\n" + "="*70)
    print("TEST 3: Verify K-Fold Scalers are Different (No Data Leakage)")
    print("="*70)

    # Load raw data
    with open(temp_files_path + 'traintest_data4.pkl', 'rb') as f:
        X_train, X_test, period_train, period_test, y_train, y_test, feature_cols, train_names, _ = pickle.load(f)

    from sklearn.model_selection import GroupKFold

    # Setup k-fold
    k = 3
    train_names = np.asarray(train_names)
    unique_clusters = np.unique(train_names)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_clusters)

    cluster_to_order = {c: i for i, c in enumerate(unique_clusters)}
    groups = np.array([cluster_to_order[c] for c in train_names])

    gkf = GroupKFold(n_splits=k)

    scalers = []
    y_means = []

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        # Get fold data
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        period_fold_train = period_train[train_idx] if period_train is not None else None
        period_fold_val = period_train[val_idx] if period_train is not None else None

        # Apply preprocessing
        _, _, _, _, _, _, scaler, y_mean, _ = apply_preprocessing(
            X_fold_train, X_fold_val,
            period_fold_train, period_fold_val,
            y_fold_train, y_fold_val
        )

        scalers.append(scaler)
        y_means.append(y_mean)

        print(f"\nFold {fold_id + 1}:")
        print(f"  Scaler mean (first 3 features): {scaler.mean_[:3]}")
        print(f"  Scaler std (first 3 features): {scaler.scale_[:3]}")
        print(f"  y_mean: {y_mean:.4f}")

    # Check that scalers are different
    scalers_differ = False
    for i in range(len(scalers) - 1):
        diff = np.abs(scalers[i].mean_ - scalers[i+1].mean_).max()
        if diff > 1e-6:
            scalers_differ = True
            print(f"\n{check_color(True)} Fold {i} and Fold {i+1} have different scalers (max diff: {diff:.6f})")

    # Check that y_means are different
    y_means_differ = len(set([round(ym, 4) for ym in y_means])) > 1

    print(f"\n{check_color(scalers_differ)} Scalers differ across folds (proving independence)")
    print(f"{check_color(y_means_differ)} Y-means differ across folds (proving independence)")

    return scalers_differ and y_means_differ


def test_4_validation_independence():
    """Test 4: Verify validation data doesn't influence training scaler"""
    print("\n" + "="*70)
    print("TEST 4: Verify Validation Data Independence")
    print("="*70)

    # Create synthetic test data
    np.random.seed(42)
    torch.manual_seed(42)

    # Training data: mean=10, std=2
    X_train_syn = torch.tensor(np.random.normal(10, 2, (100, 5)), dtype=torch.float32)
    y_train_syn = torch.tensor(np.random.normal(8, 1, (100, 1)), dtype=torch.float32)

    # Validation data: mean=100, std=20 (very different!)
    X_val_syn = torch.tensor(np.random.normal(100, 20, (20, 5)), dtype=torch.float32)
    y_val_syn = torch.tensor(np.random.normal(80, 10, (20, 1)), dtype=torch.float32)

    # Apply preprocessing
    X_train_scaled, X_val_scaled, _, _, y_train_dm, y_val_dm, scaler, y_mean, _ = apply_preprocessing(
        X_train_syn, X_val_syn,
        None, None,
        y_train_syn, y_val_syn
    )

    # Check that scaler was fit on training data only
    # If validation data influenced the scaler, scaler.mean_ would be closer to 55 (average of 10 and 100)
    scaler_mean_correct = np.abs(scaler.mean_ - 10).max() < 1.0  # Should be ~10, not ~55

    print(f"\nSynthetic Data Test:")
    print(f"  Training mean: {X_train_syn.mean(dim=0).numpy()}")
    print(f"  Validation mean: {X_val_syn.mean(dim=0).numpy()}")
    print(f"  Scaler mean: {scaler.mean_}")
    print(f"{check_color(scaler_mean_correct)} Scaler fit on training data only (not influenced by validation)")

    # Check y_mean
    y_mean_correct = abs(y_mean - 8.0) < 0.5  # Should be ~8, not ~44
    print(f"\n  Training y mean: {y_train_syn.mean().item():.4f}")
    print(f"  Validation y mean: {y_val_syn.mean().item():.4f}")
    print(f"  y_mean value: {y_mean:.4f}")
    print(f"{check_color(y_mean_correct)} y_mean computed from training data only")

    return scaler_mean_correct and y_mean_correct


def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("DATA LEAKAGE FIX VERIFICATION")
    print("="*70)
    print("\nThis script verifies that the data leakage fix is working correctly.")
    print("You should run 0_ProcessDataset.ipynb first to regenerate the data files.")

    # Check if data file exists
    data_file = Path(temp_files_path + 'traintest_data4.pkl')
    if not data_file.exists():
        print(f"\n❌ ERROR: Data file not found at {data_file}")
        print("   Please run 0_ProcessDataset.ipynb first to generate the data.")
        return

    results = []

    # Run all tests
    try:
        results.append(("Data is Unscaled (Raw)", test_1_data_is_unscaled()))
    except Exception as e:
        print(f"\n❌ Test 1 failed with exception: {e}")
        results.append(("Data is Unscaled (Raw)", False))

    try:
        results.append(("apply_preprocessing() Works", test_2_preprocessing_function()))
    except Exception as e:
        print(f"\n❌ Test 2 failed with exception: {e}")
        results.append(("apply_preprocessing() Works", False))

    try:
        results.append(("K-Fold Scalers Differ", test_3_kfold_scalers_differ()))
    except Exception as e:
        print(f"\n❌ Test 3 failed with exception: {e}")
        results.append(("K-Fold Scalers Differ", False))

    try:
        results.append(("Validation Independence", test_4_validation_independence()))
    except Exception as e:
        print(f"\n❌ Test 4 failed with exception: {e}")
        results.append(("Validation Independence", False))

    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "PASSED ✅" if passed else "FAILED ❌"
        print(f"{test_name:40s} {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Data leakage fix is working correctly.")
        print("\nNext steps:")
        print("  1. Run 1_HyperparamTuning.ipynb to retrain models with proper k-fold scaling")
        print("  2. Run 2_TrainBestModel.ipynb to train final model")
        print("  3. Compare validation metrics to previous runs (may be slightly different)")
    else:
        print("⚠️  SOME TESTS FAILED. Please check the output above for details.")
        print("\nCommon issues:")
        print("  - Make sure you ran 0_ProcessDataset.ipynb to regenerate data files")
        print("  - Check that Training_HelperFcns.py includes apply_preprocessing()")
        print("  - Verify all notebook cells were updated correctly")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
