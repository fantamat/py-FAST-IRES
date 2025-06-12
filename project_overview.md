# Understanding the FAST-IRES Python Implementation: A Reading Guide

## 1. Core Algorithm Components (Start Here)

Start with the fundamental numerical components that are used throughout the algorithm:

1. **Basic Utilities**:
   - `norms.py`: Implements vector norm computation
   - `cvx_check_dimension.py` and `cvx_default_dimension.py`: Handle dimension validation

2. **Core Optimization Algorithm**:
   - `FISTA_ADMM_IRES.py`: This contains the main optimization algorithm that powers FAST-IRES
   - `Proj_Flat_Hyper_Ellips.py`: Implements projection onto hyperellipsoids used in optimization

## 2. Main Analysis Pipeline Components

After understanding the core components, move to the main analysis pipelines:

3. **Seizure Analysis**:
   - `TBF_Selection_Seizure.py`: Preprocessing for seizure data analysis
   - `FIRES_Sz_Rot.py`: Main code for seizure imaging analysis

4. **Spike Analysis**:
   - `TBF_Spike_Extraction.py`: Preprocessing for spike data analysis
   - `FIRES_Spk_Rot.py`: Main code for spike imaging analysis

## 3. Connectivity Analysis

After understanding the basic analysis pipelines, explore the connectivity analysis:

5. **Patch and Connectivity Analysis**:
   - `Find_Patch.py`: Finds patches from IRES solutions
   - `Patch_Time_Course_Extractor.py`: Extracts time courses for connectivity analysis
   - `DTF.py`, `DTFvalue.py`, `DTFsigtest.py`, `DTFsigvalues.py`: Directed Transfer Function implementations for connectivity analysis

## 4. Results Extraction and Visualization

Finally, examine how results are extracted and visualized:

6. **Extracting Epileptogenic Zone (EZ)**:
   - `Frequency_Peak_Solution.py`: Analyzes frequency peaks for seizure data
   - `Connectivity_Segment_Solution.py`: Analyzes connectivity segments
   - `Spike_Peak_Solution.py`: Analyzes spike peaks

## Key Considerations for the Python Implementation

1. **Dependencies**:
   - NumPy: Core numerical operations
   - SciPy: Scientific computations
   - scikit-learn: ICA and machine learning functionality
   - Matplotlib: Visualization (replacing MATLAB's plotting)
   - MNE-Python: Possible replacement for some EEGLAB functionality

2. **Numba Acceleration**:
   - Focus on optimizing numerical operations in `FISTA_ADMM_IRES.py` and other computation-heavy functions

3. **Pending Implementations**:
   - `readlocs` function: For electrode location reading
   - `eegfilt`: EEG filtering functionality 
   - `arfit`: AR model estimation (possible replacements: statsmodels ARIMA or custom implementation)
   - Visualization routines using matplotlib for topographic maps

## Practical Workflow

To follow the actual analysis workflow in practice:

1. **For Seizure Analysis**:
   - Run `TBF_Selection_Seizure.py` for preprocessing (if needed)
   - Run `FIRES_Sz_Rot.py` for seizure imaging
   - Run `Patch_Time_Course_Extractor.py` for connectivity analysis
   - Run `Frequency_Peak_Solution.py` or `Connectivity_Segment_Solution.py` to extract epileptogenic zones

2. **For Spike Analysis**:
   - Run `TBF_Spike_Extraction.py` for preprocessing (if needed)
   - Run `FIRES_Spk_Rot.py` for spike imaging
   - Run `Spike_Peak_Solution.py` to extract irritative zones

This reading guide should help you navigate the Python implementation of the FAST-IRES algorithm in a logical manner, focusing first on the core components and then moving to more specialized analyses. The implementations follow the original MATLAB code structure while adapting to Python's idioms and numerical libraries.