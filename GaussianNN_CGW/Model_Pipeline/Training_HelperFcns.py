"""
Training Helper Functions

Contains core training loop functions for the Gaussian Neural Network:
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
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from itertools import cycle

# Import evaluation metrics from DataAnalysis_HelperFcns
from DataAnalysis_HelperFcns import mae, rmse, coverage, crps_gaussian, Loss_Components


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

def Run_Single_Batch(model, optimizer, X, y, loss_fn, Period_X=None, clip_grad_norm=250.0):
    optimizer.zero_grad()

    # Model forward
    if Period_X is not None:
        y_pred, sigma_pred = model(X, Period_X)
    else:
        y_pred, sigma_pred = model(X)

    # Loss computation
    loss = loss_fn(y_pred, y, sigma_pred)

    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN or Inf detected in loss!")
        return model, loss.item()

    loss.backward()

    # Check for NaN or Inf in gradients
    invalid_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            invalid_grad = True
            print(f"Bad gradient detected in {name}")

    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    if grad_norm_before > clip_grad_norm:
        print(f"[CLIP WARNING] Gradient norm was clipped to {clip_grad_norm:.4f} (before: {grad_norm_before:.4f})")
        #print('X', X)

    if invalid_grad:
        warnings.warn("Skipping optimizer step due to NaN or Inf in gradients.")
        print('Skipping optimizer')
    else:
        optimizer.step()

    return model, loss.item()


def Run_Single_Epoch(model, train_loader, optimizer, loss_fn, epoch_loss=0.0,
                     epoch=None, start_time=None, use_periodogram=False):
    n_batches = 0

    for track, batch in enumerate(train_loader, 1):
        if use_periodogram:
            batch_X, batch_P, batch_y = batch
            model, batch_loss = Run_Single_Batch(model, optimizer, batch_X, batch_y, loss_fn, Period_X=batch_P)
        else:
            batch_X, batch_y = batch
            model, batch_loss = Run_Single_Batch(model, optimizer, batch_X, batch_y, loss_fn)

        epoch_loss += batch_loss
        n_batches += 1

        if track % 30 == 0 and start_time is not None:
            elapsed_time = time.time() - start_time
            print(f"      Epoch {epoch}, Batch {track}, Loss: {batch_loss:.4f}")

    avg_epoch_loss = epoch_loss / n_batches
    return model, avg_epoch_loss


# ==============================================================================
# Full Training Loop
# ==============================================================================

def Train_Model(model, training_data, validation_data, params, log):
    X_train, Period_train, y_train = training_data
    X_valid, Period_valid, y_valid = validation_data

    learning_rate = params['lr']
    batch_size = int(params['batch_size'])
    n_epochs = int(params['n_epochs'])
    loss_factor = int(params['artificial_loss_weight_factor'])
    weight_decay = int(params['weight_decay'])
    use_periodogram = Period_train is not None

    if use_periodogram:
        train_dataset = TensorDataset(X_train, Period_train, y_train)
    else:
        train_dataset = TensorDataset(X_train, y_train)

    # Load it into a Data Loader to enable batch training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define Loss Fcn and Optimizer
    loss_fn = GaussianNLLLoss_ME(debug=False, factor=loss_factor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # Run single epoch with or without periodogram
        model, epoch_loss = Run_Single_Epoch(model, train_loader, optimizer, loss_fn, epoch_loss, use_periodogram=use_periodogram)
        log["epoch"].append(epoch)
        log["mean_batch_loss"].append(epoch_loss)

        ## LOG MODEL PROGRESS ##

        log = log_model_progress(log, model, training_data, validation_data)

        if epoch % 5 == 0:
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_time = avg_epoch_time * (n_epochs - epoch - 1)
            print(f"   Epoch {epoch}, Loss: {epoch_loss:.2f}, Estimated Time Left: {remaining_time/60:.2f} min")

    model.eval()
    with torch.no_grad():
        if Period_train is not None:
            y_pred, sigma_pred = model(X_train, Period_train)
        else:
            y_pred, sigma_pred = model(X_train)
        train_loss = loss_fn(y_pred, y_train, sigma_pred)

    return model, log, train_loss


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

def plot_run_log(run_log, kfold=True):
    """
    Plot training log for a single run with K-fold cross-validation results.

    Creates a 3-panel plot showing:
    1. Training loss (mean batch loss)
    2. Training loss components (error loss and sigma loss)
    3. Validation loss (total)

    Each K-fold is shown as a separate line with different colors.
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

    plt.show()
    return
