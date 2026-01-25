"""
Training Helper Functions

Contains core training loop functions for the Gaussian Neural Network:
- Preprocessing (apply_preprocessing)
- Loss function (GaussianNLLLoss_ME)
- Batch processing (Run_Single_Batch)
- Epoch processing (Run_Single_Epoch)
- Full model training (Train_Model)
- Logging utilities (log_model_progress)
- Plotting utilities (plot_run_log)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import warnings
import copy
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from itertools import cycle

# Import evaluation metrics from DataAnalysis_HelperFcns
from DataAnalysis_HelperFcns import mae, rmse, coverage, crps_gaussian, Loss_Components

# Import pipeline configuration for device and DataLoader settings
try:
    from pipeline_config import config
    DEFAULT_DEVICE = config.device
    DEFAULT_NUM_WORKERS = config.num_workers
    DEFAULT_PIN_MEMORY = config.pin_memory
except ImportError:
    DEFAULT_DEVICE = torch.device('cpu')
    DEFAULT_NUM_WORKERS = 0
    DEFAULT_PIN_MEMORY = False


# ==============================================================================
# Preprocessing Functions
# ==============================================================================

def apply_preprocessing(X_train, X_val, period_train, period_val, y_train, y_val, device=None):
    """
    Apply preprocessing (scaling, normalization) to training and validation data.

    Fit transformations on training data only, then apply to both train and val.
    This function ensures no data leakage by fitting scalers only on training data.

    Parameters:
    -----------
    X_train, X_val : torch.Tensor
        Summary statistics (N, 17) - raw, unscaled
    period_train, period_val : torch.Tensor or None
        Periodogram data (N, 1089) - raw, unnormalized
    y_train, y_val : torch.Tensor
        Target ages - raw, not de-meaned
    device : torch.device or None
        Device for validation data (training stays on CPU for DataLoader)

    Returns:
    --------
    X_train_scaled, X_val_scaled : torch.Tensor
        Scaled summary statistics (train on CPU, val on device)
    period_train_norm, period_val_norm : torch.Tensor or None
        Normalized periodograms (train on CPU, val on device)
    y_train_demeaned, y_val_demeaned : torch.Tensor
        De-meaned targets (train on CPU, val on device)
    scaler : StandardScaler
        Fitted scaler object
    y_mean : float
        Mean of training targets
    mean_peak_strength : float or None
        Mean peak strength for periodogram normalization (None if no periodogram)

    Notes:
    ------
    Training data is kept on CPU so DataLoader can use pin_memory for efficient GPU transfer.
    Only validation data is moved to device (since it's not batched).
    """
    from sklearn.preprocessing import StandardScaler

    # Default to configured device if not specified
    if device is None:
        device = DEFAULT_DEVICE

    # 1. De-mean targets using training mean
    y_mean = y_train.mean().item()
    # Training data stays on CPU for DataLoader
    y_train_demeaned = y_train - y_mean
    # Validation data goes to device (not batched)
    y_val_demeaned = (y_val - y_mean).to(device)

    # 2. Scale summary statistics
    scaler = StandardScaler()
    # Training data: CPU (for DataLoader with pin_memory)
    X_train_scaled = torch.tensor(
        scaler.fit_transform(X_train.cpu().numpy()),
        dtype=torch.float32
    )
    # Validation data: on device
    X_val_scaled = torch.tensor(
        scaler.transform(X_val.cpu().numpy()),
        dtype=torch.float32,
        device=device
    )

    # 3. Normalize periodograms (if provided)
    if period_train is not None:
        periodograms = period_train.cpu().numpy()
        mean_peak_strength = np.max(periodograms, axis=1).mean()

        # Training: keep on CPU for DataLoader
        period_train_norm = period_train / mean_peak_strength
        # Validation: move to device
        period_val_norm = (period_val / mean_peak_strength).to(device)
    else:
        period_train_norm = None
        period_val_norm = None
        mean_peak_strength = None

    return (X_train_scaled, X_val_scaled,
            period_train_norm, period_val_norm,
            y_train_demeaned, y_val_demeaned,
            scaler, y_mean, mean_peak_strength)


# ==============================================================================
# Loss Function
# ==============================================================================

class GaussianNLLLoss_ME(nn.Module):
    def __init__(self, min_sigma=1e-5, max_sigma=1e5, debug=False, factor=1):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.debug = debug
        self.weight_factor = factor

    def forward(self, y_pred, y_true, sigma_pred):
        # Convert raw sigma to strictly positive sigma using softplus
        #sigma_pred = F.softplus(raw_sigma_pred) + self.min_sigma
        #sigma_pred = torch.clamp(sigma_pred, max=self.max_sigma)
        # Sanitize predictions
        #y_pred, sigma_pred, was_clipped = sanitize_batch(y_pred, sigma_pred)

        #if was_clipped:
        #    warnings.warn("[CLIP WARNING] y_pred was clipped!")


        # Compute the Gaussian negative log-likelihood loss
        nll = 0.5 * (self.weight_factor * (((y_true - y_pred) ** 2) / (sigma_pred ** 2)) + torch.log(2*np.pi * sigma_pred**2))

        if self.debug:
            print(f"[DEBUG] y_pred range: {y_pred.min().item():.4f} to {y_pred.max().item():.4f}")
            print(f"[DEBUG] sigma_pred range: {sigma_pred.min().item():.4f} to {sigma_pred.max().item():.4f}")
            print(f"[DEBUG] NLL mean: {nll.mean().item():.4f}")
            if torch.any(torch.isnan(nll)) or torch.any(torch.isinf(nll)):
                print("[WARNING] NaN or Inf detected in loss!")

                nan_mask = torch.isnan(nll)
                inf_mask = torch.isinf(nll)
                bad_mask = nan_mask | inf_mask

                bad_indices = torch.nonzero(bad_mask, as_tuple=False).squeeze()

                for idx in bad_indices:
                    print(f"[DEBUG][BAD INDEX {idx.item()}] y_pred: {y_pred[idx].item():.4f}, "
                          f"sigma_pred: {sigma_pred[idx].item():.4f}, y_true: {y_true[idx].item():.4f}, "
                          f"NLL: {nll[idx].item()}")

        #plt.hist(nll.detach().numpy())
       # plt.show()

        return nll.mean()


def sanitize_batch(y_pred, sigma_pred, clip_sigma=(1e-7, 100.0), clip_pred=(-1000.0, 1000.0)):
    """
    Clamp sigma_pred and y_pred to avoid numerical instability.
    Also returns a flag if y_pred was actually clipped.
    """
    original_y_pred = y_pred.clone().detach()

    y_pred = torch.clamp(y_pred, min=clip_pred[0], max=clip_pred[1])
    sigma_pred = torch.clamp(sigma_pred, min=clip_sigma[0], max=clip_sigma[1])

    was_clipped = not torch.allclose(original_y_pred, y_pred)

    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        print("[SANITY CHECK] y_pred contains NaN or Inf!")

    if torch.isnan(sigma_pred).any() or torch.isinf(sigma_pred).any():
        print("[SANITY CHECK] sigma_pred contains NaN or Inf!")

    return y_pred, sigma_pred, was_clipped


# ==============================================================================
# Batch and Epoch Training Functions
# ==============================================================================

def Run_Single_Batch(model, optimizer, X, y, loss_fn, Period_X=None, clip_grad_norm=250.0, device=None,
                     batch_indices=None, cluster_names=None, epoch=None):
    optimizer.zero_grad()

    # Move batch to device if specified
    if device is not None:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if Period_X is not None:
            Period_X = Period_X.to(device, non_blocking=True)

    # Model forward
    try:
        if Period_X is not None:
            y_pred, sigma_pred = model(X, Period_X)
        else:
            y_pred, sigma_pred = model(X)
    except RuntimeError as e:
        print(f"[ERROR] Forward pass failed: {e}", flush=True)
        print(f"  X shape: {X.shape}, device: {X.device}", flush=True)
        if Period_X is not None:
            print(f"  Period_X shape: {Period_X.shape}, device: {Period_X.device}", flush=True)
        raise

    # Loss computation
    loss = loss_fn(y_pred, y, sigma_pred)

    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN or Inf detected in loss!", flush=True)
        return model, loss.item()

    # Check for extreme loss values that will corrupt the model
    loss_value = loss.item()
    # if loss_value > 100.0:  # Normal losses are 0.5-3.0, anything >100 is catastrophic
    #     warnings.warn(
    #         f"EXTREME LOSS DETECTED: {loss_value:.2f} - SKIPPING THIS BATCH\n"
    #         f"  y_pred range: [{y_pred.min().item():.2e}, {y_pred.max().item():.2e}]\n"
    #         f"  y_true range: [{y.min().item():.2e}, {y.max().item():.2e}]\n"
    #         f"  sigma_pred range: [{sigma_pred.min().item():.2e}, {sigma_pred.max().item():.2e}]\n"
    #         f"DATA QUALITY ISSUE: Your training data contains extreme outliers."
    #     )
    #     print(f"[SKIP WARNING] Skipping batch with extreme loss {loss_value:.2f} to prevent model corruption", flush=True)
    #     # Return without updating model weights
    #     return model, loss_value

    try:
        loss.backward()
    except RuntimeError as e:
        print(f"[ERROR] backward() failed: {e}", flush=True)
        print(f"  Loss value: {loss.item()}", flush=True)
        print(f"  y_pred range: [{y_pred.min().item():.2e}, {y_pred.max().item():.2e}]", flush=True)
        print(f"  sigma_pred range: [{sigma_pred.min().item():.2e}, {sigma_pred.max().item():.2e}]", flush=True)
        raise

    # Check for NaN or Inf in gradients
    invalid_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            invalid_grad = True
            print(f"Bad gradient detected in {name}")

    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    if grad_norm_before > clip_grad_norm and epoch is not None and epoch > 5:
        print(f"\t[CLIP WARNING] Grad norm clipped to {clip_grad_norm:.0f} (was {grad_norm_before:.0f})", flush=True)

        # Compute per-sample residuals to find problematic data points
        residuals = (y - y_pred.detach()).abs()
        normalized_residuals = residuals / (sigma_pred.detach() + 1e-8)

        # Get indices of top 2 highest normalized residuals
        top_k = min(2, X.shape[0])
        _, top_indices = torch.topk(normalized_residuals.squeeze(), top_k)

        for i in top_indices:
            # Get cluster name if available
            if batch_indices is not None and cluster_names is not None:
                orig_idx = batch_indices[i].item()
                cluster = cluster_names[orig_idx]
            else:
                orig_idx = i.item()
                cluster = "N/A"

            print(f"\t\t  Sample {orig_idx} ({cluster}): y_true={y[i].item():.4f}, "
                  f"y_pred={y_pred[i].item():.4f}, sigma={sigma_pred[i].item():.4f}, "
                  f"norm_resid={normalized_residuals[i].item():.4f}",
                  flush=True)

    if invalid_grad:
        warnings.warn("Skipping optimizer step due to NaN or Inf in gradients.")
        print('Skipping optimizer')
    else:
        try:
            optimizer.step()
        except RuntimeError as e:
            print(f"[ERROR] optimizer.step() failed: {e}", flush=True)
            print(f"  Gradient norm before clipping: {grad_norm_before:.4f}", flush=True)
            raise

    return model, loss_value


def Run_Single_Epoch(model, train_loader, optimizer, loss_fn, epoch_loss=0.0,
                     epoch=None, start_time=None, use_periodogram=False, device=None,
                     cluster_names=None):
    n_batches = 0
    from datetime import datetime

    print(f"[EPOCH START] Epoch {epoch}, Total batches: {len(train_loader)}", flush=True)

    for track, batch in enumerate(train_loader, 1):
        if track % 10 == 0 or track == 1:
            print(f"[BATCH] Epoch {epoch}, Batch {track}/{len(train_loader)}", flush=True)
        if use_periodogram:
            batch_X, batch_P, batch_y, batch_indices = batch
            model, batch_loss = Run_Single_Batch(model, optimizer, batch_X, batch_y, loss_fn, Period_X=batch_P, device=device,
                                                  batch_indices=batch_indices, cluster_names=cluster_names, epoch=epoch)
        else:
            batch_X, batch_y, batch_indices = batch
            model, batch_loss = Run_Single_Batch(model, optimizer, batch_X, batch_y, loss_fn, device=device,
                                                  batch_indices=batch_indices, cluster_names=cluster_names, epoch=epoch)

        epoch_loss += batch_loss
        n_batches += 1

        # NOTE: Reduced from every 30 to every 10 batches for crash debugging. Revert if performance impact.
        if track % 10 == 0 and start_time is not None:
            elapsed_time = time.time() - start_time
            wall_time = datetime.now().strftime("%H:%M:%S")
            print(f"      [{wall_time}] Epoch {epoch}, Batch {track}/{len(train_loader)}, Loss: {batch_loss:.4f}, Elapsed: {elapsed_time/60:.1f}min", flush=True)

    avg_epoch_loss = epoch_loss / n_batches
    return model, avg_epoch_loss


# ==============================================================================
# Full Training Loop
# ==============================================================================

def Train_Model(model, training_data, validation_data, params, log, save_best_checkpoint=False, device=None,
                cluster_names=None):
    """
    Train a model with optional best checkpoint saving.

    Parameters:
    -----------
    model : nn.Module
        The model to train (should already be on the correct device)
    training_data : tuple
        (X_train, Period_train, y_train)
    validation_data : tuple
        (X_valid, Period_valid, y_valid)
    params : dict
        Training parameters
    log : dict
        Training log dictionary
    save_best_checkpoint : bool, optional
        If True, tracks and returns best model checkpoint based on smoothed validation loss.
        If False, only returns final model (backward compatible).
        Default: False
    device : torch.device or None
        Device for training (default: CPU)

    Returns:
    --------
    If save_best_checkpoint=False (default):
        model, log, train_loss
    If save_best_checkpoint=True:
        model, log, train_loss, best_model_state, best_epoch
    """
    # Default to configured device if not specified
    if device is None:
        device = DEFAULT_DEVICE

    X_train, Period_train, y_train = training_data
    X_valid, Period_valid, y_valid = validation_data

    # Keep original training data on CPU for DataLoader
    # Create GPU copies for evaluation
    X_train_eval = X_train.to(device)
    Period_train_eval = Period_train.to(device) if Period_train is not None else None
    y_train_eval = y_train.to(device)

    # Move validation data to device once (it's reused every epoch)
    X_valid = X_valid.to(device)
    y_valid = y_valid.to(device)
    if Period_valid is not None:
        Period_valid = Period_valid.to(device)

    learning_rate = params['lr']
    batch_size = int(params['batch_size'])
    n_epochs = int(params['n_epochs'])
    loss_factor = float(params['artificial_loss_weight_factor'])  # FIXED: was int, now float
    weight_decay = float(params['weight_decay'])  # FIXED: was int (converted 1e-06 to 0), now float
    use_periodogram = Period_train is not None

    # Create index tensor for tracking original sample indices (for gradient debug output)
    indices = torch.arange(len(X_train))

    if use_periodogram:
        train_dataset = TensorDataset(X_train, Period_train, y_train, indices)
    else:
        train_dataset = TensorDataset(X_train, y_train, indices)

    # Configure DataLoader using settings from pipeline_config
    # GPU: pin_memory=True for faster host->GPU transfer
    # CPU: pin_memory=False
    # num_workers from config (0 on WSL due to multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY
    )

    # Define Loss Fcn and Optimizer
    loss_fn = GaussianNLLLoss_ME(debug=False, factor=loss_factor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Best model tracking (only if requested)
    if save_best_checkpoint:
        best_model_state = None
        best_epoch = 0
        best_smoothed_loss = float('inf')
        smoothing_window = 3  # Smooth over 3 epochs
        recent_val_losses = []  # Store recent validation losses for smoothing

    start_time = time.time()
    from datetime import datetime

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        # Run single epoch with or without periodogram
        model, epoch_loss = Run_Single_Epoch(model, train_loader, optimizer, loss_fn, epoch_loss,
                                            epoch=epoch, start_time=start_time, use_periodogram=use_periodogram,
                                            device=device, cluster_names=cluster_names)
        log["epoch"].append(epoch)
        log["mean_batch_loss"].append(epoch_loss)

        ## LOG MODEL PROGRESS ##
        # Use GPU copies for evaluation (created once at start)
        training_data_eval = (X_train_eval, Period_train_eval, y_train_eval)
        validation_data_eval = (X_valid, Period_valid, y_valid)
        log = log_model_progress(log, model, training_data_eval, validation_data_eval)

        # Best model checkpoint tracking
        if save_best_checkpoint:
            # Get current validation loss for best model tracking
            current_val_loss = log["valid_Loss"][-1]
            recent_val_losses.append(current_val_loss)

            # Keep only the last 'smoothing_window' losses
            if len(recent_val_losses) > smoothing_window:
                recent_val_losses.pop(0)

            # Compute smoothed validation loss (average over recent epochs)
            smoothed_val_loss = sum(recent_val_losses) / len(recent_val_losses)

            # Update best model if we found a new minimum smoothed validation loss
            if smoothed_val_loss < best_smoothed_loss:
                best_smoothed_loss = smoothed_val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0:
            elapsed_time = time.time() - start_time
            epoch_time = time.time() - epoch_start
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_time = avg_epoch_time * (n_epochs - epoch - 1)
            wall_time = datetime.now().strftime("%H:%M:%S")
            val_loss = log["valid_Loss"][-1] if log["valid_Loss"] else 0.0
            print(f"   [{wall_time}] Epoch {epoch}/{n_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"      Epoch time: {epoch_time:.1f}s, Avg: {avg_epoch_time:.1f}s, Total elapsed: {elapsed_time/60:.1f}min, Est. remaining: {remaining_time/60:.1f}min")

    # Final training loss (from last epoch) - use GPU evaluation copies
    model.eval()
    with torch.no_grad():
        if Period_train_eval is not None:
            y_pred, sigma_pred = model(X_train_eval, Period_train_eval)
        else:
            y_pred, sigma_pred = model(X_train_eval)
        final_train_loss = loss_fn(y_pred, y_train_eval, sigma_pred)

    # Return based on checkpoint mode
    if save_best_checkpoint:
        print(f"\n   📌 Best model checkpoint: Epoch {best_epoch} (smoothed val_loss={best_smoothed_loss:.4f})")
        return model, log, final_train_loss, best_model_state, best_epoch
    else:
        return model, log, final_train_loss


# ==============================================================================
# Logging Functions
# ==============================================================================

def log_model_progress(log, model, train_data, valid_data):
    model.eval()
    data_label = ['train_', 'valid_']
    for i, data in enumerate([train_data, valid_data]):
        label = data_label[i]


        y_true = data[2]

        with torch.no_grad():
            if data[1] is not None:
                y_pred, sigma_pred = model(data[0], data[1])
            else:
                y_pred, sigma_pred = model(data[0])

        #print(np.shape(y_true), np.shape(y_pred), np.shape(sigma_pred), np.shape(data[0]), np.shape(data[1]))
        err, sigma, tot = Loss_Components(y_true, y_pred, sigma_pred)
        log[label+"errLoss"].append(err)
        log[label+"sigmaLoss"].append(sigma)
        log[label+"Loss"].append(tot)
        log[label+"MAE" ].append(mae(y_true, y_pred))
        log[label+"RMSE"].append(rmse(y_true, y_pred))
        log[label+"median_sigma"].append(torch.median(sigma_pred).item())
        log[label+"mean_sigma"].append(torch.mean(sigma_pred).item())
        log[label+"Coverage68"].append(coverage(y_true, y_pred, sigma_pred, num_sigma=1.0))
        log[label+"Coverage95"].append(coverage(y_true, y_pred, sigma_pred, num_sigma=2.0))
        log[label+"CRPS"].append(crps_gaussian(y_true, y_pred, sigma_pred))

    return log


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_run_log(run_log, kfold=True, save_path=None):
    """
    Plot training log for a single run with K-fold cross-validation results.

    Creates a 3-panel plot showing:
    1. Training loss (mean batch loss)
    2. Training loss components (error loss and sigma loss)
    3. Validation loss (total)

    Each K-fold is shown as a separate line with different colors.

    Parameters:
    -----------
    run_log : dict
        Training log dictionary containing params and fold results
    kfold : bool
        Whether to plot K-fold results (default: True)
    save_path : str, optional
        If provided, saves the figure to this path instead of showing it.
        Should include the filename (e.g., 'path/to/run_0.png')
    """
    params = run_log['params']

    fig = plt.figure(figsize=(12,10))
    #fig.suptitle(f"Training Log for Model: {model_name}", fontsize=16)
    fig.suptitle(str(params), fontsize=8)

    gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])
    ax1 = fig.add_subplot(gs[0])  # Training loss
    ax2 = fig.add_subplot(gs[1])  # Train components
    ax3 = fig.add_subplot(gs[2])  # Valid loss

    colors = plt.cm.tab10.colors
    color_cycle = cycle(colors)
    legend_handles = []

    if kfold:
        items = [(f'k={fold_id}', run_log[fold_id]["log"])
                for fold_id in range(params['num_kfolds'])]
    else:
        items = [('final model', run_log["log"])]

    for label, log in items:
        color = next(color_cycle)

        # Total training loss
        ax1.plot(log["epoch"], log["mean_batch_loss"], label=label, color=color)

        # Train components
        ax2.plot(log["epoch"], log["train_errLoss"], color=color, linestyle='solid', label=label)
        ax2.plot(log["epoch"], np.asarray(log["train_sigmaLoss"]), color=color, linestyle="--")

        # Validation components
        ax3.plot(
            log["epoch"],
            np.asarray(log["valid_errLoss"]) + np.asarray(log["valid_sigmaLoss"]),
            color=color,
            linestyle='solid',
            label=label
        )

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

    for ax in [ax1, ax2, ax3]:
        ax.legend()

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        # Save to file instead of showing
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()

    return
