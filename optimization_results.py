# -*- coding: utf-8 -*-


#pip install optuna==2.10.1

import joblib
import optuna

# Load the study object using joblib
study = joblib.load("orient-pixels_per_cell-cells_per_block40.pkl")

# Manually set the _directions attribute
study._directions = ('MAXIMIZE')  # Replace with the appropriate direction

# Access the best trial information
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

optuna.visualization.plot_optimization_history(study)

