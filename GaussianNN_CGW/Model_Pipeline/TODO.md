
# Notes to Self for future implimentation

## Pipeline ToDos

- Run a large grid of models (1_HyperparamTuning.ipynb)
- Impliment early stoping (2_HyperparamTuning.ipynb)
- Write Test code notebook (3_ModelEvaluation.ipynb)
    - Core Performance Plots:
        - Predicted vs True Age scatter (with error bars)
        - Residuals vs True Age
        - Residuals vs Predicted Age
        - Residuals distribution histogram
    - Uncertainty Calibration Plots:
        - Z-score distribution with QQ-plot and KS test
        - Coverage calibration curve (empirical vs theoretical)
        - Uncertainty vs Absolute Error scatter
        - Predicted uncertainty distribution
    - Performance Breakdown Plots:
        - Error metrics by age bin (MAE, RMSE across young/intermediate/old)
        - Coverage by age bin
        - Performance by individual cluster (best/worst)
    - Summary Statistics:
        - Comprehensive metrics table (MAE, RMSE, Coverage, CRPS, KS test)
    - Optional Advanced:
        - MC Dropout uncertainty decomposition
        - Feature importance analysis

## General Notes For improvement

- Cleaner data
- Identify/Visualize model performance on idividual clusters
    - this may help with data cleaning
- Idenfity/visualize feature importance and model weights.
    - this would help if we ever wish to reduce model complexity

## Goals for End of December
- Get a draft-'final' model 
- Evaluate that model's performance


## Notes to Self:

### Towards reducing high-variance in k-fold validation loss and/or the crazyness of the loss-landscape

- Plot which clusters are causing validation loss explosion
- Run with more k-folds    <---- Finished
- Run with smaller model   <---- Finished
- perform some level of feature engineering:
    - Reduce feature space (eg. PCA, but simpler feature cutting steps could also work)
    - Data cleaning: Removing spurious clusters, removing high noise clusters, removing outliers?
- this is also somewhat expected given our small dataset and large feature space.

Summary of solutions
    - Dimensionalty reduction (basic removal of params/PCA/Periodogram)
    - Stronger Regularization (Smaller model, higher dropout, L1/L2, early stopping)
    - Cluster-based normalization (Normalize periodograms per cluster before training) <-- do we already do this?
    - Change periodogram approach (eg. CNN)

### Would be Nice to have:
- Quantify how the model interacts with individual parameters (ie. stats and/or periodogram)
- Quantify how the model interacts with individual clusters

On that front could make a 3.2_ModelBehaviorAnalysis quantifying these things.
This would look like:
A) Visualize Model Weights. 
    - See which ones have large amplitude. 
    - I have some old code that does this.
    - Could be nice to reduce feature space?
B) Visualize how individual clusters interaction
    1) Are there individual clusters which cause wild training issues
        - Not 100% sure how to approach 
    2) Identify how the model performs on clusters.
        - Does it get some clusters right and some wrong (eg. strugles with ultra young/old clusters).
        - Or does it get some LC from some clusters right and other LC wrong
            - difference in those LCs?


