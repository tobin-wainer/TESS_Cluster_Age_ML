# TESS_Cluster_Age_ML

Machine learning techniques to predict stellar cluster ages from integrated (spatially unresolved) light curves using TESS photometric data.

## Project Overview

This repository contains a complete pipeline for training neural networks to estimate the ages of stellar clusters based on their integrated light curves from the Transiting Exoplanet Survey Satellite (TESS). The approach leverages variability in integrated light curves, which contain information about the stellar populations within clusters that correlates with age.

### Scientific Motivation

Stellar cluster ages are fundamental to understanding stellar evolution, galactic archaeology, and calibrating age-dating techniques. Gold Standard age estimation methods rely on color-magnitude diagrams and isochrone fitting, which require resolved stellar photometry. This project explores using integrated light curves as an alternative age indicator, which is particularly valuable for:
- Distant or crowded clusters where individual stars cannot be resolved
- Large-scale surveys where integrated photometry is readily available
- Breaking the age-extinction degenercy of current integrated light methods.

### Data Processing Pipeline

1. **Raw Light Curves** (`light_curves/`): TESS Full Frame Image (FFI) light curves for 348 stellar clusters across three galaxies:
   - 124 Milky Way (MW) clusters
   - 118 Large Magellanic Cloud (LMC) clusters
   - 106 Small Magellanic Cloud (SMC) clusters

2. **Feature Extraction** (Root directory notebooks): Transform raw light curves into summary statistics and periodogram features
   - `Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector.ipynb`: Generates summary statistics
   - `Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector_wPeriodogram.ipynb`: Adds periodogram-based features

3. **Processed Datasets** (`data/`): FITS tables containing extracted features for ML training

### Model Approaches

The repository implements multiple ML architectures:

1. **Gaussian Neural Networks** (`GaussianNN_CGW/`): Primary approach using networks with Gaussian likelihood outputs
   - Progressive development from V1 through V4
   - Integration of periodogram features
   - K-fold cross-validation implementation
   - Modular pipeline structure (`Model_Pipeline/`)

2. **Standard Neural Networks**: Data-augmented neural network implementations (`Data_Augmented_NN_v7_running_several_times.ipynb`)

3. **Random Forests**: Ensemble tree-based approach (`Data_Augmented_RF_v7_running_several_times.ipynb`)

4. **Regression Models**: Traditional regression techniques with sector-stitched weighting

## Key Files

### Root Directory

- `Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector.ipynb`: Generates summary statistics data tables from raw light curves
- `Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector_wPeriodogram.ipynb`: Extended version that adds periodogram features (added by CGW)
- `Data_Augmented_NN_v7_running_several_times.ipynb`: Neural network training with data augmentation
- `Data_Augmented_RF_v7_running_several_times.ipynb`: Random forest model training
- `Using_Regression_with_sector_stitched_weighting.ipynb`: Regression-based age prediction
- `Run_Regression_v2.py`: Python script for regression model execution

### GaussianNN_CGW Directory

Contains progressive development of Gaussian neural network models:
- `CGW_SimpleNN_V1.ipynb` through `CGW_SimpleNN_V4_wPeriodigram_wKFold.ipynb`: Iterative model improvements
This is the developmental precursor to the Model_Pipeline directory.

### Model_Pipeline Directory

The core model directory where a NN is trained on the the summary statistics+periodogram data from (`data/`).
Modular pipeline for Gaussian NN training:
- `0_ProcessDataset.ipynb`: Converts summary statistics and periodograms into PyTorch-compatible datasets
- `1_HyperparamTuning-Copy1.ipynb`: Hyperparameter optimization for Gaussian NN
- `2_TrainBestModel.ipynb`: Training final model with optimal hyperparameters
- `GaussianNN_wPeriodogram.py`: Neural network class definitions
- `*_PlottingFcns.py`: Visualization and analysis helper functions

## Getting Started

**Note**: Notebook paths will need to be updated to match your local directory structure.

1. Raw TESS light curves are stored in `light_curves/MW/`, `light_curves/LMC/`, and `light_curves/SMC/`
2. Run feature extraction notebooks to generate summary statistics
3. Processed datasets will be saved to `data/` directory
4. Use model training notebooks/pipeline to train age prediction models

## Data

Light curve summary statistics for data-augmented light curves are available in the `data/` folder. 
Raw light curve FITS files are organized by galaxy in the `light_curves/` folder. 


Here's the directory tree for the TESS_Cluster_Age_ML repository: (Last Updated Dec 2, 2025)
  .
  ├── README.md
  ├── Run_Regression_v2.py
  ├── Data_Augmented_NN_v7_running_several_times.ipynb
  ├── Data_Augmented_RF_v7_running_several_times.ipynb
  ├── Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector.ipynb
  ├── Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector_wPeriodograms.ipynb
  ├── Using_Regression_with_sector_stitched_weighting.ipynb
  │
  ├── data/
  │   ├── readme.txt
  │   ├── Bica_Cut_down.fits
  │   ├── Glatt_Cut_down.fits
  │   ├── Use_MW.fits
  │   ├── Stats_Table_For_all_Clusters.fits
  │   ├── Stats_Table_For_all_Clusters_all_sectors.fits
  │   ├── Stats_Table_For_all_Clusters_all_sectors_w_entropy.fits
  │   ├── Stats_Table_For_all_Clusters_all_sectors_w_entropy_wPeriodogram.fits
  │   ├── Stats_Table_AllClusters_HalfSectors_wPeriodogram.fits
  │   ├── Stats_Table_For_Data_augmented_lcs_v2.fits
  │   └── Stats_Table_For_Sector_Stitched.fits
  │
  ├── light_curves/
  │   ├── readme
  │   ├── MW/           (124 .fits files)
  │   ├── LMC/          (118 .fits files)
  │   └── SMC/          (106 .fits files)
  │
  └── GaussianNN_CGW/
      ├── CGW_SimpleNN_V1.ipynb
      ├── CGW_SimpleNN_V2.ipynb
      ├── CGW_SimpleNN_V3_wPeriodigram.ipynb
      ├── CGW_SimpleNN_V4_wPeriodigram_wKFold.ipynb
      │
      └── Model_Pipeline/
          ├── 0_ProcessDataset.ipynb                          <-- Turns raw LC summary statistics and periodogram into NN friendly pytorch dataset
          ├── 1_HyperparamTuning.ipynb                        <-- Loads in dataset and trains a gaussian NN
          ├── 2_TrainBestModel.ipynb                          <-- New code in progress
          ├── OG_Notebook.ipynb                               <-- Old Notebook which was expanded into this pipeline
          ├── GaussianNN_wPeriodogram.py                      <-- Contains NN class object code
          ├── DataAnalysis_HelperFcns.py                      <-- Plotting Helper Functions
          ├── DropoutStat_PlottingFcns.py                     <-- Plotting Helper Functions
          ├── Kfold_PlottingFcns.py                           <-- Plotting Helper Functions
          ├── SingleFold_PlottingFcns.py                      <-- Plotting Helper Functions
          ├── untitled.txt
          ├── untitled1.txt
          ├── untitled2.txt
          │
          ├── HelperFcns/
          │   └── VisualizeDataset.py
          │
          └── TempFiles/
              ├── traintest_data.pkl
              ├── traintest_data2.pkl
              └── ModelGridLogs/



### OLD Information
Repo for the working Directory of ML techniques to predict Star cluster ages from integrated light curves

Light Curve summary statistics for data augmented light curves are available in the data folder. Trying to get the actual light curve fits files in the light_curves folder

Jupyter Notebooks are in the home directory. Paths will need to be updated

Files:

Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector.ipynb
	- was used to generate the summary statistics data tables
Reading_in_LCs_Data_Augmentation_and_Getting_stats_Every_Sector_wPeriodogram.ipynb 
	- Modified version of Tobin's original version
	- was used to generate the summary statistics + period-gram data tables (added by CGW)
GussianNN_CGW
	- Folder where I, CGW, am adding the gaussian NN notebooks I have been working on. 
