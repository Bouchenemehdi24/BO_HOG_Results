## Results Notebook

This notebook provides a comprehensive guide to reproduce the results of a face recognition system. The system uses Histogram of Oriented Gradients (HOG) parameters and image size that were optimized by Bayesian optimization as described in the paper by Bouchene, Mohammed Mehdi, "Bayesian Optimization of Histogram of Oriented Gradients (HOG) Parameters for Facial Recognition". The system is trained and evaluated on three different datasets: ORL, Extended Yale B, and AR Face Database.

## FacePix.py
The `facepix.py` script allows you to reproduce the results specifically on the **FacePix dataset**. Here's what it does:

1. **Data Preparation**:
   - Downloads and extracts the FacePix dataset.
   - Preprocesses the images (resizing, normalization, etc.).

2. **Feature Extraction**:
   - Computes HOG features for each face image.
   - Optionally applies dimensionality reduction (PCA).

3. **Model Training and Evaluation**:
   - Trains a Ridge Classifier on the HOG features.
   - Evaluates the model's performance using cross-validation.

4. **Visualization and Analysis**:
   - Displays random face images.
   - Visualizes the HOG features.
   - Reports accuracy and other metrics.

## Dependencies

The notebook requires the following libraries:
- `zipfile`
- `rarfile`
- `gdown`
- `imutils`
- `sklearn`
- `matplotlib`
- `cv2`
- `numpy`
- `skimage`

You can install them using pip:
```python
!pip install zipfile rarfile gdown imutils sklearn matplotlib opencv-python numpy scikit-image
```

## Datasets

The notebook uses three datasets for training and evaluation:

1. **ORL Dataset**: Contains 400 images of 40 subjects. Each subject has 10 different images taken at different times, varying the lighting, facial expressions, and facial details. The images are in grayscale and have a resolution of 92x112 pixels.

2. **Extended Yale B Database**: Contains 2452 frontal-face images of 38 individuals. The images were taken under various laboratory-controlled lighting conditions. The images are in PNG format and have a resolution of 192x168 pixels.

3. **AR Face Database**: Contains over 4,000 color images of 126 people's faces (70 men and 56 women). The images were taken at two different sessions, separated by two weeks. The images show variations in facial expression, illumination, and occlusion (sun glasses and scarf). The images are in JPEG format and have a resolution of 768x576 pixels.

## Functions

The notebook defines several functions to handle different tasks:

1. `unzip_dataset(dataset_path, extract_path)`: Unzips a dataset.

2. `unrar_file(path_to_rar_file, path_to_extract_to)`: Unrars a file.

3. `display_random_images(imgorl, num_images=5)`: Displays a specified number of randomly chosen images from a given list of image paths.

4. `extract_features_and_labels(face_image_files)`: Extracts features and labels from the face images.

5. `convert_to_gray(face_image_path)`: Reads an image from the given path and converts it to grayscale.

6. `resize_image(face_image, size)`: Resizes an image to the given size.

7. `train_and_evaluate_model(hog_features, face_labels, test_size=0.3, random_state=0)`: Trains and evaluates the model.

## Usage

To use this notebook, you need to download the datasets and extract them. Then, you can display some random images from the datasets, extract features and labels from the face images, and finally train and evaluate the model.

## Author

This notebook was authored by Dr. Bouchene Mohammed Mehdi. If you have any questions or require further clarification, feel free to reach out to him at bouchenemahdi@gmail.com.

## Citation

If you find this notebook beneficial for your research or work, please consider citing the following paper:

Bouchene, M. M. (2023). Bayesian Optimization of Histogram of Oriented Gradients (HOG) Parameters for Facial Recognition. Available at SSRN: https://ssrn.com/abstract=4506361 or http://dx.doi.org/10.2139/ssrn.4506361

Here's the BibTeX entry for easy citation:

```bibtex
@article{bouchene2023bayesian,
  title={Bayesian Optimization of Histogram of Oriented Gradients (HOG) Parameters for Facial Recognition},
  author={Bouchene, Mohammed Mehdi},
  journal={Available at SSRN 4506361},
  year={2023}
}
