

def visualize_feature_distributions(data, feature_cols, plot_full_range=False):
    fig, axs = plt.subplots(4, 5, figsize=(15,15))
    
    for index, col in enumerate(feature_cols):
        i = index // 5
        j = index % 5
        #print(i, j, col)
        print(col)
        median = np.nanmedian(data[:, index])  # Ignore NaNs
        mad = np.nanmedian(np.abs(data[:, index] - median))  # Compute MAD
        scale_factor = 1.4826  # Approximate conversion to std deviation
        lower_bound = median - 4 * scale_factor * mad
        upper_bound = median + 4 * scale_factor * mad
        
        if plot_full_range:
            axs[i, j].hist(data[:, index], bins=1000)#, range=(lower_bound, upper_bound))
            #axs[i, j].set_xlim([lower_bound, upper_bound])
        else:
            axs[i, j].hist(data[:, index], bins=1000, range=(lower_bound, upper_bound))
            axs[i, j].set_xlim([lower_bound, upper_bound])

        axs[i, j].set_title(col)
        axs[i, j].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def visualize_mass_bin_distribution(y_train, y_test):
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs[0, 0].set_title('Distribution of Ages in our samples')
    
    axs[0, 0].hist(y_train, bins=2)
    axs[0, 0].set_ylabel('test data, bins=2')
    
    axs[0, 1].hist(y_train, bins=3)
    axs[0, 1].set_ylabel('test data, bins=3')
    
    axs[1, 0].hist(y_train, bins=5)
    axs[1, 0].set_ylabel('test data, bins=5')
    
    axs[1, 1].hist(y_train, bins=7)
    axs[1, 1].set_ylabel('test data, bins=7')
    
    axs[2, 1].hist(y_test)
    axs[2, 1].set_ylabel('y_test, auto-bining')
    

    axs[2, 1].set_xlabel('de-meaned log stellar Age')
    axs[2, 0].set_xlabel('de-meaned log stellar Age')

    for axs_row in axs:
        for ax in axs_row:
            ax.set_xlim([-1.6, 2.2])
    plt.show()
    return

def visualize_summary_stats(data, data_y, feature_cols, mad_bounds=4):
    
    n_features = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n_features / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axs = np.atleast_1d(axs)  # safe for 1-row cases
    
    for index, col in enumerate(feature_cols):
        ax = axs.flat[index]
        
        #print(i, j, col)
        median = np.nanmedian(data[:, index])  # Ignore NaNs
        mad = np.nanmedian(np.abs(data[:, index] - median))  # Compute MAD
        
        ax.scatter(data[:, index], data_y, s=0.2, alpha=0.5)
        ax.set_title(col)
        #axs[i, j].set_yscale('log')
        if mad_bounds!=None:
            scale_factor = 1.4826  # Approximate conversion to std deviation
            lower_bound = median - mad_bounds * scale_factor * mad
            upper_bound = median + mad_bounds * scale_factor * mad
            ax.set_xlim([lower_bound, upper_bound])
        ax.set_xlabel(col)
        ax.set_ylabel('log Cluster Age')
    
    plt.tight_layout()
    plt.show()





import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(X, feature_names=None, figsize=(10, 8), annot=False):
    # Convert to NumPy if needed
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)

    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Plot using seaborn
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, xticklabels=feature_names, yticklabels=feature_names, 
                cmap="coolwarm", center=0, annot=annot, fmt=".2f", square=True, cbar_kws={"shrink": 0.75})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return corr_matrix

