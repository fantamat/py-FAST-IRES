"""
ARfit: A Python package for the estimation of parameters and eigenmodes of multivariate autoregressive models.

This package implements algorithms for the estimation of parameters and eigenmodes of multivariate
autoregressive (AR) models, diagnostic checking of fitted AR models, and analysis of eigenmodes.

The algorithms are described in the following papers:

Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of multivariate 
autoregressive models. ACM Trans. Math. Softw., 27, 27-57.

Schneider, T., and A. Neumaier, 2001: Algorithm 808: ARfit — A Matlab package for the estimation
of parameters and eigenmodes of multivariate autoregressive models. ACM Trans. Math. Softw., 27, 58-65.

This is a Python port of the original MATLAB ARfit package by Tapio Schneider and Arnold Neumaier,
enhanced with Numba acceleration and DTF connectivity integration for the FAST-IRES project.
"""

import numpy as np
from scipy import linalg

# Import all modules
from .acf import acf
from .adjph import adjph
from .arconf import arconf
from .ardem import run_demo
from .arfit import arfit
from .armode import armode
from .arord import arord
from .arqr import arqr
from .arres import arres
from .arsim import arsim
from .tquant import tquant

# Import new modules for FAST-IRES enhancements
try:
    from .numba_optimized import qr_factorization, compute_sbc_fpe, compute_eigendecomposition, compute_periods_damping
    from .dtf_integration import prepare_ar_for_dtf, compute_dtf
    from .optimized_dtf import optimized_dtf, compute_dtf_from_ar_params
    from .visualization import (
        plot_eigenvalues, 
        plot_frequency_response,
        plot_ar_residuals
    )
    
    __all__ = [
        'acf', 'adjph', 'arconf', 'run_demo', 'arfit', 'armode', 
        'arord', 'arqr', 'arres', 'arsim', 'tquant',
        'prepare_ar_for_dtf', 'compute_dtf', 'optimized_dtf',
        'plot_eigenvalues', 'plot_frequency_response', 'plot_ar_residuals'
    ]
except ImportError as e:
    # Some enhancements might not be available
    print(f"Warning: Some ARfit enhancements are not available: {e}")
    
    __all__ = [
        'acf', 'adjph', 'arconf', 'run_demo', 'arfit', 'armode', 
        'arord', 'arqr', 'arres', 'arsim', 'tquant'
    ]
from .tquant import tquant

# Define package metadata
__title__ = "ARfit"
__version__ = "1.0.0"
__description__ = "Python port of ARfit: Multivariate Autoregressive Model Fitting"
__license__ = "ACM Software License Agreement"

# Define package exports
__all__ = [
    'acf',
    'adjph',
    'arconf',
    'arfit',
    'armode',
    'arord',
    'arqr',
    'arres',
    'arsim',
    'tquant',
    'run_demo',
]
