
# Notes to Self for future implimentation

## General Pipeline ToDos

- Test the model-retrain notebook (2_TrainBestModel.ipynb)
- Write Test code notebook (3_ModelEvaluation.ipynb)
- Move scalertransform code into k-fold notebook to prevent data leakage.

## General Notes For improvement

- Cleaner data
- Identify/Visualize model performance on idividual clusters
    - this may help with data cleaning
- Idenfity/visualize feature importance and model weights.
    - this would help if we ever wish to reduce model complexity

### Towards reducing high-variance in k-fold validation loss and/or the crazyness of the loss-landscape

- Plot which clusters are causing validation loss explosion
- Run with more k-folds?
- Run with smaller model
- perform some level of feature engineering:
    - Reduce feature space (eg. PCA, but simpler feature cutting steps could also work)
    - Data cleaning: Removing spurious clusters, removing high noise clusters, removing outliers?
- this is also somewhat expected given our small dataset and large feature space.

Summary of solutions
    - Dimensionalty reduction (basic removal of params/PCA/Periodogram)
    - Stronger Regularization (Smaller model, higher dropout, L1/L2, early stopping)
    - Cluster-based normalization (Normalize periodograms per cluster before training) <-- do we already do this?
    - Change periodogram approach (eg. CNN)

## Goals for End of December
- Get a draft-'final' model 
- Evaluate that model's performance

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


