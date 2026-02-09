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



# This is the nueral network model.
class DualInputNN(nn.Module):
    '''
    This is the primary Nueral Network Model.
    Depending on the inputs at definition, 
        it will/wont use the periodigram data
        it will/wont be a CNN
    So it has multiple possible architectures.
    '''
    def __init__(self,
                 summary_dim,
                 periodogram_dim=0,
                 x1=64,
                 dropout_prob=0.3,
                 use_periodogram=True,
                 periodogram_use_cnn=False,
                 learn_sigma=True):  # NEW FLAG
        super(DualInputNN, self).__init__()
        ''' 
        summary_dim = # of summary statistics
        periodogram_dim = # of frequency bins on periodigrams
        x1 = hidden layer size
        dropout_prob = dropout probability (for regularization and MC-Dropout testing)
        use_periodogram = enable/disable using the periodgram in the model
        periodogram_use_cnn = enable/disable using CNN vs a linear model for the periodgram.
        learn_sigma = enable/disable having the model predict uncertainties (if False will output a constant uncertainty of 0.1)
        ''' 
        
        self.use_periodogram = use_periodogram
        self.periodogram_use_cnn = periodogram_use_cnn
        self.learn_sigma = learn_sigma  # NEW ATTR

        # Summary stats branch (always active)
        self.summary_branch = nn.Sequential(
            nn.Linear(summary_dim, x1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Periodogram branch (optional)
        if self.use_periodogram:
            if self.periodogram_use_cnn:
                self.periodogram_branch = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=5, padding=2), #if input is (#Batches, 1, len_periodogram), 
                                                                #  output is (#Batches, 16, len_periodogram) (because kernel=5 while padding=2) each of the 16 is the same convoution with different random weights.
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(16, 32, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(p=dropout_prob)
                    # Add regularization Layer
                )
                periodogram_out_dim = 32
            else:
                self.periodogram_branch = nn.Sequential(
                    nn.Linear(periodogram_dim, x1),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_prob)
                )
                periodogram_out_dim = x1
        else:
            periodogram_out_dim = 0

        combined_dim = x1 + periodogram_out_dim
        
        self.fc_mean = nn.Linear(combined_dim, 1)

        if self.learn_sigma:
            self.fc_log_sigma = nn.Linear(combined_dim, 1)  # only used if learn_sigma
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, summary_x, periodogram_x=None):
        summary_feat = self.summary_branch(summary_x)

        if self.use_periodogram:
            if self.periodogram_use_cnn:
                if periodogram_x.ndim == 2:
                    periodogram_x = periodogram_x.unsqueeze(1)
                periodogram_feat = self.periodogram_branch(periodogram_x)
            else:
                periodogram_feat = self.periodogram_branch(periodogram_x)
            combined = torch.cat([summary_feat, periodogram_feat], dim=1)
        else:
            combined = summary_feat

        y_pred = self.fc_mean(combined)

        if self.learn_sigma:
            log_sigma_pred = self.fc_log_sigma(combined)
            sigma_pred = torch.exp(log_sigma_pred)
        else:
            sigma_pred = torch.ones_like(y_pred) * 0.001  # dummy sigma for MSE mode

        return y_pred, sigma_pred



