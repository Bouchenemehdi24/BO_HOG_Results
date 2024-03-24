# Author: Bouchene Mohammed Mehdi

from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.transform import resize

def show_hog_images(image, params):
    """
    This function takes an image and a list of parameters, and displays the Histogram of Oriented Gradients (HOG) for the image.
    
    Parameters:
    image (ndarray): The input image.
    params (list): A list of dictionaries. Each dictionary contains the parameters for the HOG feature extraction.
    
    The dictionary has the following keys:
    - 'size' (tuple): The desired size of the image.
    - 'orientations' (int): The number of orientation bins.
    - 'pixels_per_cell' (tuple): Size (in pixels) of a cell.
    - 'cells_per_block' (tuple): Number of cells in each block.
    - 'transform_sqrt' (bool): Apply power law compression to normalize the image before processing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    for i, param in enumerate(params):
        # Resize the image
        image_resized = resize(image, param['size'])
        fd, hog_image = hog(image_resized, orientations=param['orientations'], pixels_per_cell=param['pixels_per_cell'], cells_per_block=param['cells_per_block'], transform_sqrt = param['transform_sqrt'], visualize=True)
        ax = axes[i//2, i%2]
        ax.axis('off')  # Turn off axis
        # Normalize the size of the image for display
        hog_image_resized = resize(hog_image, (356, 356))
        ax.imshow(hog_image_resized, cmap=plt.cm.gray, aspect='auto')  # Set aspect to auto for automatic resizing
        ax.set_title(f'({chr(65+i)})')

    plt.tight_layout()
    plt.show()
