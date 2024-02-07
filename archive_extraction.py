# -*- coding: utf-8 -*-


"""
Author: Dr Bouchene Mohammed Mehdi

This module provides functions to unzip a dataset and unrar a file.
"""

import zipfile
import rarfile

def unzip_dataset(dataset_path, extract_path):
    """
    This function unzips a dataset.

    Parameters:
    dataset_path (str): The path to the zipped dataset.
    extract_path (str): The path where the dataset should be extracted to.

    Returns:
    None
    """
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def unrar_file(path_to_rar_file, path_to_extract_to):
    """
    This function unrars a file.

    Parameters:
    path_to_rar_file (str): The path to the rar file.
    path_to_extract_to (str): The path where the rar file should be extracted to.

    Returns:
    None
    """
    with rarfile.RarFile(path_to_rar_file) as rf:
        rf.extractall(path_to_extract_to)