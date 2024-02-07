# Author: Bouchene Mohammed Mehdi

import matplotlib.pyplot as plt
import cv2
import random

def display_random_images(imgorl, num_images=5):
    """
    Display a specified number of randomly chosen images from a given list of image paths.

    Parameters:
    imgorl (list): A list of image paths.
    num_images (int): The number of images to display. Default is 5.

    Returns:
    None
    """
    # Randomly select images
    random_images = random.sample(imgorl, num_images)

    # Create a figure to contain the subplots
    fig = plt.figure(figsize=(12, 12))

    for i, img_path in enumerate(random_images):
        # Read the image
        img = cv2.imread(img_path)
        # Convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Add a subplot for this image
        ax = fig.add_subplot(1, num_images, i+1)
        ax.imshow(img)
        plt.axis('off')

    plt.show()
