



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
import math
import copy
import warnings



#### FUNCTION TO HELP TEST THE MODEL DURING TRAINING ####
from scipy.stats import norm
import numpy as np


def mae(y_true, y_pred):# Mean Absolute Error  # Robust to Outliers
    return torch.mean(torch.abs(y_true - y_pred))

def rmse(y_true, y_pred): # Root Mean Squared Error # Sensitive to large errors — shows how “bad” outliers are
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mean_normalized_error(y_true, y_pred, sigma):           # This is useful to check if errors are consistent with predicted uncertainties
    return torch.mean(torch.abs((y_true - y_pred) / sigma))

def Loss_Components(y_true, y_pred, sigma_pred):
    err_loss = torch.mean(((y_true - y_pred) ** 2) / (sigma_pred ** 2))
    sigma_loss = torch.mean(2 * torch.log(sigma_pred))
    return err_loss, sigma_loss, err_loss+sigma_loss

def coverage(y_true, y_pred, sigma_pred, num_sigma=1.0):
    lower = y_pred - num_sigma * sigma_pred
    upper = y_pred + num_sigma * sigma_pred
    inside = (y_true >= lower) & (y_true <= upper)
    return torch.mean(inside.float()).item()

def crps_gaussian(y_true, y_pred, sigma_pred):
    """Compute CRPS for Gaussian predictive distribution (Torch version)."""
    z = (y_true - y_pred) / sigma_pred

    # normal cdf approximation with erf
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)

    crps = sigma_pred * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    return torch.mean(crps).item()

####################################################################################



def coverage(y_true, y_pred, sigma_pred, num_sigma=1.0):
    lower = y_pred - num_sigma * sigma_pred
    upper = y_pred + num_sigma * sigma_pred
    inside = (y_true >= lower) & (y_true <= upper)
    return torch.mean(inside.float()).item()




def test_coverage_extended():
    print("=== Fixed Test Cases ===")
    # Test case 1: Perfect prediction
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    sigma_pred = torch.tensor([0.1, 0.1, 0.1])
    print("Test 1:", coverage(y_true, y_pred, sigma_pred))  # expect 1.0

    # Test case 2: All outside
    y_true = torch.tensor([1.2, 2.2, 3.2])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    sigma_pred = torch.tensor([0.1, 0.1, 0.1])
    print("Test 2:", coverage(y_true, y_pred, sigma_pred))  # expect 0.0

    # Test case 3: Mixed
    y_true = torch.tensor([1.0, 2.1, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    sigma_pred = torch.tensor([0.2, 0.2, 0.2])
    print("Test 3:", coverage(y_true, y_pred, sigma_pred))  # expect ~0.67

    # Test case 4: Large sigma
    y_true = torch.tensor([1.5, 2.5, 3.5])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    sigma_pred = torch.tensor([1.0, 1.0, 1.0])
    print("Test 4:", coverage(y_true, y_pred, sigma_pred, num_sigma=2.0))  # expect 1.0

    # More mixed cases
    print("\n=== Additional Mixed Cases ===")
    cases = [
        ([1.0, 2.2, 2.9, 4.1], [1.0, 2.0, 3.0, 4.0], [0.2, 0.3, 0.1, 0.5], 1.0),
        ([5.0, 5.5, 6.0], [5.0, 5.0, 5.0], [0.2, 0.5, 1.0], 1.0),
        ([0.9, 2.0, 2.1], [1.0, 2.0, 2.0], [0.1, 0.2, 0.1], 1.0),
    ]
    for i, (yt, yp, sp, ns) in enumerate(cases, start=5):
        yt = torch.tensor(yt)
        yp = torch.tensor(yp)
        sp = torch.tensor(sp)
        print(f"Test {i}:", coverage(yt, yp, sp, num_sigma=ns))

    print("\n=== Randomized NumPy Tests ===")
    np.random.seed(42)
    for i in range(3):
        N = 1000
        y_pred = np.random.randn(N)
        sigma_pred = np.abs(np.random.randn(N)) * 0.5 + 0.1
        y_true = y_pred + sigma_pred * np.random.randn(N)  # Gaussian noise
        cov = coverage(
            torch.tensor(y_true),
            torch.tensor(y_pred),
            torch.tensor(sigma_pred),
            num_sigma=1.0
        )
        print(f"Random Test {i+1}: coverage ≈ {cov:.2f} (expect ≈ 0.68 for 1σ Gaussian)")

    print("\n=== Randomized Higher Sigma ===")
    for i in range(3):
        N = 1000
        y_pred = np.random.randn(N)
        sigma_pred = np.abs(np.random.randn(N)) * 0.5 + 0.1
        y_true = y_pred + sigma_pred * np.random.randn(N)
        cov = coverage(
            torch.tensor(y_true),
            torch.tensor(y_pred),
            torch.tensor(sigma_pred),
            num_sigma=2.0
        )
        print(f"Random Test {i+1} (2σ): coverage ≈ {cov:.2f} (expect ≈ 0.95)")



