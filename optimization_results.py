# -*- coding: utf-8 -*-

"""
Author: Dr. Bouchene Mohammed Mehdi

This script loads the results of an optimization process from a file, prints the best result and the parameters that yielded it, 
and visualizes the optimization history. 

The optimization process was conducted using the Optuna library with the Tree-structured Parzen Estimator (TPE) algorithm. 
The parameters being optimized are the Histogram of Oriented Gradients (HOG) feature descriptor and the dimensions of resized images. 

The best trial is the one that yielded the highest value of the objective function, 
which represents the best 5-fold cross-validation accuracy of Ridge classifier on ORL dataset, 
indicating the most optimal parameters found during the optimization process. 

The visualization helps in understanding how the optimization progressed over time. 
Please ensure that the required dependencies are installed and that the file "orient-pixels_per_cell-cells_per_block40.pkl" is in the 
correct location for this script to run successfully. 

!pip install optuna==2.10.1
!pip install joblib

"""

# Import necessary libraries
import joblib
import optuna

# Load the study object using joblib
# The study object contains the results of the optimization process
# The file "orient-pixels_per_cell-cells_per_block40.pkl" is assumed to be in the same directory
study = joblib.load("orient-pixels_per_cell-cells_per_block40.pkl")

# Manually set the _directions attribute
# This attribute determines whether the optimization is to maximize or minimize the objective function
# In this case, we are maximizing
study._directions = ('MAXIMIZE')

# Access the best trial information
# The best trial is the one that yielded the highest value of the objective function
# In this case, the objective function represents the best 5-fold cross-validation accuracy
print("Best trial until now:")
print(" Value: ", study.best_trial.value)  # Print the best value

# Print the parameters that yielded the best value
# The parameters being optimized are the Histogram of Oriented Gradients (HOG) feature descriptor iimplemented in skimage
# and the dimensions of resized images
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Visualize the optimization history
# This function creates a plot of the objective function value over the course of the optimization process
# The optimization process was conducted using the Optuna library with the Tree-structured Parzen Estimator (TPE) algorithm
optuna.visualization.plot_optimization_history(study)
