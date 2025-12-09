# Model Pipeline

This directory contains a modular pipeline for training Gaussian Neural Networks to predict stellar cluster ages from TESS light curve features. The models output both age predictions and associated uncertainties.

## Overview

The pipeline implements a probabilistic neural network that:
- Predicts cluster ages from summary statistics and periodogram features
- Outputs Gaussian distributions (mean and sigma) for uncertainty quantification
- Supports flexible architectures (with/without periodograms, CNN/Linear periodogram processing)
- Uses K-fold cross-validation with cluster-based splitting
- Tracks comprehensive evaluation metrics

## Pipeline Structure

### Step 0: Data Processing (`0_ProcessDataset.ipynb`)

**Purpose**: Transform raw FITS tables into PyTorch-compatible datasets

**Inputs**:
- `../../data/Stats_Table_*_wPeriodogram.fits` - FITS tables with summary statistics and periodogram data

**Outputs**:
- `TempFiles/traintest_data2.pkl` - Pickled dataset containing:
  - `X_train`, `X_test` - Summary statistics (17 features: RMS, std, MAD, skewness, power spectrum metrics, entropy, etc.)
  - `period_train`, `period_test` - Periodogram data (1089 frequency bins)
  - `y_train`, `y_test` - Log-scaled cluster ages
  - `train_cluster_names`, `test_cluster_names` - Cluster identifiers for group-based K-fold splitting
  - `SCALER` - StandardScaler for input normalization
  - `y_mean` - Mean age for denormalization

**Key Operations**:
- Loads FITS tables with cluster data
- Extracts 17 summary statistics and periodogram features
- Applies log-scaling to ages: `log10(age/1e6)`
- Standardizes features using sklearn StandardScaler
- Splits data into train/test sets
- Exports as pickle for efficient loading

### Step 1: Hyperparameter Tuning (`1_HyperparamTuning-Copy1.ipynb`)

**Purpose**: Train and evaluate models across hyperparameter grids using K-fold cross-validation

**Workflow**:

1. **Load Data**: Imports processed dataset from Step 0
2. **Define Model Architecture**: Configures `DualInputNN` with options for:
   - Summary statistics branch (always active)
   - Optional periodogram branch (CNN or Linear)
   - Learned or fixed uncertainty estimation

3. **Configure Hyperparameter Grid**:
   ```python
   params = {
       'use_periodogram': [True/False],
       'use_cnn': [True/False],
       'learn_sigma': [True/False],
       'n_layers': [1, 2, 3],
       'Layer1_Size': [16, 32, 64],
       'dropout_prob': [0.2, 0.3, 0.4],
       'lr': [1e-3, 1e-4],
       'batch_size': [16, 32, 100],
       'n_epochs': [50, 100],
       'num_kfolds': [5]
   }
   ```

4. **K-Fold Cross-Validation**:
   - Uses `GroupKFold` to split by cluster names (prevents data leakage from same cluster appearing in train/validation)
   - Trains separate model for each fold
   - Tracks metrics across all folds

5. **Training Loop**: For each hyperparameter combination:
   - Initializes fresh model
   - Trains using Adam optimizer with Gaussian NLL loss
   - Logs extensive metrics every epoch:
     - **Loss Components**: Error term, sigma term, total loss
     - **Accuracy**: MAE, RMSE
     - **Uncertainty Calibration**: Coverage@68%, Coverage@95%, CRPS
     - **Sigma Statistics**: Median, mean predicted uncertainty

6. **Model Evaluation**:
   - Plots training/validation loss evolution
   - Visualizes all metrics vs epoch with K-fold statistics (median � MAD)
   - Ranks models by composite score (weighted combination of loss, coverage, and error metrics)
   - Generates color-coded performance plots

7. **Advanced Analysis**:
   - **MC Dropout**: Enables dropout at test time to sample prediction distributions
   - **Uncertainty Decomposition**: Separates epistemic (MC dropout std) from aleatoric (predicted sigma) uncertainty

**Outputs**:
- `TempFiles/ModelGridLogs/*.pkl` - Saved training logs with models and metrics
- Comprehensive plots of model performance across hyperparameter space

### Step 2: Train Best Model (`2_TrainBestModel.ipynb`)

**Status**: In development

**Purpose**: Retrain optimal model on full training set (no K-fold) using best hyperparameters identified in Step 1

## Core Components

### Neural Network Architecture (`GaussianNN_wPeriodogram.py`)

**`DualInputNN` Class**:
```python
DualInputNN(
    summary_dim,              # Number of summary statistics (17)
    periodogram_dim=1089,     # Periodogram frequency bins
    x1=64,                    # Hidden layer size
    dropout_prob=0.3,         # Dropout probability
    use_periodogram=True,     # Enable periodogram branch
    periodogram_use_cnn=False,# Use CNN (True) or Linear (False) for periodogram
    learn_sigma=True          # Learn uncertainty (True) or use fixed sigma (False)
)
```

**Architecture Options**:
1. **Summary Statistics Only**: Single branch with Linear � ReLU � Dropout
2. **With Periodogram (Linear)**: Dual branches concatenated before final layers
3. **With Periodogram (CNN)**: Uses Conv1D � MaxPool � AdaptiveAvgPool for periodogram processing

**Outputs**: `(y_pred, sigma_pred)` - Gaussian distribution parameters

**Loss Function**: `GaussianNLLLoss_ME`
```python
loss = 0.5 * (((y_true - y_pred)^2 / sigma_pred^2) + log(2� * sigma_pred^2))
```

### Evaluation Metrics (`DataAnalysis_HelperFcns.py`)

**Accuracy Metrics**:
- `mae`: Mean Absolute Error - Robust to outliers
- `rmse`: Root Mean Squared Error - Sensitive to large errors
- `mean_normalized_error`: Checks if errors scale with predicted uncertainty

**Uncertainty Calibration**:
- `coverage(y_true, y_pred, sigma, num_sigma)`: Fraction of true values within N-sigma intervals
  - Ideal: ~68% for 1-sigma, ~95% for 2-sigma
- `z_scores`: Standardized residuals (should follow N(0,1) if well-calibrated)
- `z_score_ks_test`: Kolmogorov-Smirnov test for normality of z-scores

**Probabilistic Scores**:
- `crps_gaussian`: Continuous Ranked Probability Score - Proper scoring rule for probabilistic predictions
- `Loss_Components`: Decomposes loss into error term and sigma term

### Plotting Functions

**K-Fold Visualization** (`Kfold_PlottingFcns.py`):
- `plot_trainval_loss_evolution`: Training/validation loss across epochs for all K-folds
- `plot_all_metric_evolution`: All metrics vs epoch with median � MAD error bands across folds

**Single Model Analysis** (`SingleFold_PlottingFcns.py`):
- `plot_Ypred_v_Ytrue_withErrs`: Prediction vs truth scatter with error bars
- `SummaryStats`: Comprehensive statistical summary with KS test, coverage, etc.
- `PlotSummaryStats`: Multi-panel visualization of all metrics
- `Plot_2d_Statistics`: Heatmap of performance metrics across 2D hyperparameter grid
- `permutation_importance`: Feature importance via permutation (identifies critical features)

## Model Selection Strategy

The hyperparameter tuning notebook implements a systematic model selection process:

1. **Coarse Grid Search**: Test architectural decisions (with/without periodogram, CNN vs Linear)
2. **Refined Grid**: Explore learning rate, batch size, dropout, layer size
3. **Dropout Analysis**: Find optimal dropout rate for best architecture
4. **Final Training**: Train best model configuration on full dataset

**Ranking Criteria**: Models are ranked by composite score:
```python
score = w_loss * validation_loss
      + w_cov68 * |coverage_68 - 0.68|
      + w_cov95 * |coverage_95 - 0.95|
      + w_err1 * RMSE
      + w_err2 * MAE
```

## Key Features

### Group-Based K-Fold Cross-Validation

Uses `GroupKFold` with cluster names as groups to prevent data leakage:
- Ensures all observations from a single cluster stay in the same fold
- More realistic evaluation (tests generalization to unseen clusters)
- Avoids artificially inflated performance from sector-to-sector correlations

### Uncertainty Quantification

**Aleatoric Uncertainty** (inherent data noise):
- Predicted by the network's sigma output
- Learned during training via Gaussian NLL loss

**Epistemic Uncertainty** (model uncertainty):
- Estimated via MC Dropout at test time
- Sample prediction distribution by running model multiple times with dropout enabled
- Useful for identifying out-of-distribution or ambiguous inputs

### Gradient Clipping and Numerical Stability

- Clips gradients to prevent explosion during training
- Sanitizes predictions and sigmas to avoid NaN/Inf in loss computation
- Monitors for numerical issues and skips updates if detected

## Dependencies

```python
torch, torch.nn, torch.optim
numpy, scipy
matplotlib, seaborn
sklearn (StandardScaler, GroupKFold, ParameterGrid)
astropy (fits tables)
```

## Usage Example

```python
# Load processed data
with open('TempFiles/traintest_data2.pkl', 'rb') as f:
    X_train, X_test, period_train, period_test, y_train, y_test, \
    feature_cols, SCALER, y_mean, train_cluster_names, test_cluster_names = pickle.load(f)

# Initialize model
model = DualInputNN(
    summary_dim=17,
    periodogram_dim=1089,
    x1=32,
    dropout_prob=0.3,
    use_periodogram=True,
    periodogram_use_cnn=False,
    learn_sigma=True
)

# Train with K-fold cross-validation
all_runs_log = kfold_cross_validation(
    X_train, period_train, train_cluster_names, y_train,
    model_class=DualInputNN,
    Params=best_params,
    k=5
)

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred, sigma_pred = model(X_test, period_test)
```

## Directory Contents

```
Model_Pipeline/
   0_ProcessDataset.ipynb              # Data loading and preprocessing
   1_HyperparamTuning.ipynb      # Hyperparameter search with K-fold CV
   2_TrainBestModel.ipynb              # Final model training (in progress)
   OG_Notebook.ipynb                   # Original exploratory notebook

   GaussianNN_wPeriodogram.py          # Neural network class definition
   DataAnalysis_HelperFcns.py          # Evaluation metrics
   Kfold_PlottingFcns.py               # K-fold visualization functions
   SingleFold_PlottingFcns.py          # Detailed model analysis plots
   DropoutStat_PlottingFcns.py         # MC Dropout analysis (not yet reviewed)

   HelperFcns/
      VisualizeDataset.py             # Dataset visualization utilities

   TempFiles/
       traintest_data.pkl              # Processed datasets
       traintest_data2.pkl
       ModelGridLogs/                  # Saved training runs
           *.pkl
```

## Notes

- Training logs can be large (contains all models and metrics across parameter grid)
- Use pickle files for quick experimentation without retraining
- MC Dropout adds computational cost but provides valuable uncertainty estimates
- Coverage metrics are the most important for evaluating uncertainty calibration
- Well-calibrated models should achieve ~68% coverage at 1-sigma and ~95% at 2-sigma

