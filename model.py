# -*- coding: utf-8 -*-
# Import the necessary modules for data processing and machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import RidgeClassifier
import time

def train_and_evaluate_model(hog_features, face_labels, test_size=0.3, random_state=0):
    """
    Train and evaluate the model.

    Args:
        hog_features (np.array): The HOG features for training.
        face_labels (np.array): The face labels for training.
        test_size (float): The size of the test set.
        random_state (int): The seed for the random number generator.

    Returns:
        None
    """
    # Split the data into training and testing sets, stratifying by face labels to ensure balanced classes
    trainX, testX, trainY, testY = train_test_split(hog_features, face_labels, stratify=face_labels, random_state=random_state)

    # Print the number of images used in training and testing
    print("[INFO] The number of images used in training: " + str(trainX.shape[0]))
    print("[INFO] The number of images used in testing: " + str(testX.shape[0]))

    # Record the start time of the training process
    start_time = time.time()

    # Create an instance of the Ridge classifier with default hyperparameters
    # The Ridge classifier is a linear model that uses ridge regression to fit the data
    # It is efficient and scalable for large datasets
    classifier = RidgeClassifier()

    # Train the classifier model on the training data
    model = classifier.fit(trainX, trainY)

    # Print the elapsed time of the training process
    print("--- %s seconds ---" % (time.time() - start_time))

    # Make predictions on the test data and show a classification report
    print("[INFO] Evaluating...")
    predictions = model.predict(testX)
    report = classification_report(testY, predictions, digits=4)

    print(report)

    # Compute the accuracy score of the test predictions
    accp = accuracy_score(testY, predictions)
    # Compute the number of correctly classified faces in the test set
    numbrcorr = accuracy_score(testY, predictions, normalize=False)

    # Make predictions on the training data and compute the accuracy score
    tr_preds = model.predict(trainX)
    trcc = accuracy_score(trainY, tr_preds)

    # Print the train and test accuracy scores and the number of correctly classified faces
    print("Train Accuracy: %.4f%%" % (trcc * 100.0))
    print("Test Accuracy: %.4f%%" % (accp * 100.0))
    print("Number of correctly classified faces: " + str(numbrcorr))
