"""
Initialization file for the custom package.
This package contains various utilities and modules for data processing,
model evaluation, feature engineering, and more.

Modules Included:
- `selfDef`: Contains custom layers and functions for deep learning models.
- `metadataUtils`: Provides functions for processing and extracting metadata features from text and videos.
- `matrixUtils`: Contains utilities for model evaluation, plotting confusion matrices, and performance metrics.
- `utils`: Includes functions for data oversampling, normalization, and handling imbalanced datasets.
- `standardUtils`: Provides standard data processing utilities used throughout the package.

Imports are set up for ease of use when the package is imported.
"""

# Importing all necessary modules for the package
from . import selfDef  # DIY components for custom layers and loss functions
from . import metadataUtils  # Utilities for processing metadata from text and video data
from . import matrixUtils  # Utilities for model evaluation and plotting confusion matrices
from . import utils  # Functions for data handling and oversampling
from . import standardUtils  # Standard utilities for general data processing

# Optionally, expose common functions/classes directly from submodules
from .selfDef import Attention, coAttention_para, myLossFunc, tagOffSet, zero_padding
from .metadataUtils import metadata, get_wordcount, get_wordcount_obj
from .matrixUtils import plot_confusion_matrix, model_predict
from .utils import oversampling, NormalizeData, get_popularity_score, measure_popularity_score
from .standardUtils import *  # Import everything from standardUtils, if needed

__all__ = [
    'selfDef', 'metadataUtils', 'matrixUtils', 'utils', 'standardUtils',
    'Attention', 'coAttention_para', 'myLossFunc', 'tagOffSet', 'zero_padding',
    'metadata', 'get_wordcount', 'get_wordcount_obj',
    'plot_confusion_matrix', 'model_predict',
    'oversampling', 'NormalizeData', 'get_popularity_score', 'measure_popularity_score'
]
