import numpy as np
from imblearn.over_sampling import SMOTE

IMAGE_FEATURE_SIZE = 2048
TEXT_FEATURE_SIZE = 768
METADATA_SIZE = 8  # Assuming METADATA_SIZE is used as part of the feature size calculation

def oversampling(text_shuffle_train, image_shuffle_train, y_vrank_shuffle_train, y_lrank_shuffle_train, y_shuffle_train):
    """
    Performs oversampling using SMOTE to balance the training dataset.

    Args:
    - text_shuffle_train: Text feature data for training (numpy array).
    - image_shuffle_train: Image feature data for training (numpy array).
    - y_vrank_shuffle_train: Additional ranking features for training (numpy array).
    - y_lrank_shuffle_train: Additional ranking features for training (numpy array).
    - y_shuffle_train: Target labels for training (numpy array).

    Returns:
    - text_over: Oversampled text features.
    - image_over: Oversampled image features.
    - y_vrank_over: Oversampled y_vrank features.
    - y_lrank_over: Oversampled y_lrank features.
    - y_over: Oversampled target labels.
    """
    # Assign input data to internal variables
    _text = text_shuffle_train
    _image = image_shuffle_train
    _y_vrank = y_vrank_shuffle_train
    _y_lrank = y_lrank_shuffle_train
    _y = y_shuffle_train

    # Combine all features into a single feature set for SMOTE
    # Ensure proper concatenation of image, text, and ranking features
    image2 = _image  # Use the original image without squeezing
    rank_pd = _y_vrank
    rank_pd2 = _y_lrank

    # Concatenate text, image, and rank features along the correct axis
    feature = np.concatenate((_text, image2, rank_pd, rank_pd2), axis=1)

    # Define the oversampling strategy and apply SMOTE
    sampling_strategy = 1
    smt = SMOTE(sampling_strategy=sampling_strategy)
    feature_over, y_over = smt.fit_resample(feature, _y)

    # Extract oversampled features back into their respective components
    text_over = feature_over[:, :TEXT_FEATURE_SIZE]
    image_over = feature_over[:, TEXT_FEATURE_SIZE:TEXT_FEATURE_SIZE + IMAGE_FEATURE_SIZE]
    y_vrank_over = feature_over[:, TEXT_FEATURE_SIZE + IMAGE_FEATURE_SIZE:
                                TEXT_FEATURE_SIZE + IMAGE_FEATURE_SIZE + TEXT_FEATURE_SIZE]
    y_lrank_over = feature_over[:, TEXT_FEATURE_SIZE + IMAGE_FEATURE_SIZE + TEXT_FEATURE_SIZE:]

    # Return oversampled datasets
    return text_over, image_over, y_vrank_over, y_lrank_over, y_over
