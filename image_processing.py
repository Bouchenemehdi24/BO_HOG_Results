# -*- coding: utf-8 -*-
# Author: Bouchene Mohammed Mehdi

# Import the cv2 module for image processing
import cv2

def convert_to_gray(face_image_path):
    """
    Read an image from the given path and convert it to grayscale.

    Args:
        face_image_path (str): The path of the face image file.

    Returns:
        face_gray_image: The grayscale face image.
    """
    # Read the image from the file
    face_image = cv2.imread(face_image_path)
    # Convert the image to grayscale using the cv2.cvtColor() function
    face_gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # Return the grayscale image
    return face_gray_image

def resize_image(face_image, size):
    """
    Resize an image to the given size.

    Args:
        face_image: The input face image.
        size (int): The desired size.

    Returns:
        resized_face_image: The resized face image.
    """
    # Resize the image using the cv2.resize() function
    # The first argument is the input image
    # The second argument is a tuple of the desired width and height in pixels
    # The third argument is the interpolation method, which can be cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, or cv2.INTER_NEAREST
    # We use cv2.INTER_AREA for shrinking the image, as it is the preferred method for image decimation
    resized_face_image = cv2.resize(face_image, (size, size), interpolation=cv2.INTER_AREA)
    # Return the resized image
    return resized_face_image
