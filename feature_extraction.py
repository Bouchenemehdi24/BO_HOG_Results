# -*- coding: utf-8 -*-
# Author: Bouchene Mohammed Mehdi

# Import the os module for file operations
import os
# Import the numpy module for numerical computations
import numpy as np
# Import the hog function from skimage for feature extraction
from skimage.feature import hog
# Import custom functions for image processing
from image_processing import convert_to_gray, resize_image

def extract_features_and_labels(face_image_files):
    """
    Extract features and labels from the face images.

    Args:
        face_image_files (list): The list of face image files.

    Returns:
        hog_features (np.array): The extracted HOG features.
        labels (np.array): The extracted labels.
    """
    # Initialize an empty list to store the HOG features
    hog_features = []
    # Initialize an empty list to store the labels
    labels = []

    # Print the number of face images in the dataset
    print('Number of images in the dataset:', len(face_image_files))

    # Loop through each face image file
    for file in face_image_files:
        # Convert the face image to grayscale
        face_image = convert_to_gray(file)
        # Resize the face image to 362x362 pixels
        face_image = resize_image(face_image, 362)
        # Extract the HOG features from the face image using the optimized parameters
        # The orientations argument specifies the number of orientation bins for the histogram
        # The pixels_per_cell argument specifies the size of the cell for which the histogram is computed
        # The cells_per_block argument specifies the number of cells in each block
        # The transform_sqrt argument applies a power law compression to normalize the image before processing
        features = hog(face_image, orientations=25, pixels_per_cell=(36, 36), cells_per_block=(2, 2), transform_sqrt=False)
        # Extract the label from the file name
        label = file.split(os.path.sep)[-2]
        # Append the label to the labels list
        labels.append(label)
        # Append the features to the HOG features list
        hog_features.append(features)

    # Return the HOG features and labels as numpy arrays
    return np.array(hog_features), np.array(labels)
