Implementation of PCA with ANN for Face Recognition

This project implements a Face Recognition System using Principal Component Analysis (PCA) and an Artificial Neural Network (ANN) classifier.
The system follows the classical Eigenfaces + Neural Network approach and is built using Python, NumPy, SciPy, OpenCV, and Scikit-learn (ANN).
Accuracy Improvement

The face recognition system was evaluated by experimenting with different values of k (number of principal components). The accuracy significantly improved as the feature extraction and classification pipeline progressed from traditional PCA to PCA combined with ANN.

1)Baseline Accuracy (PCA + Euclidean Distance) — ~62%

Using only PCA with Euclidean distance as the classifier resulted in a modest accuracy of 62%, due to linear separability limits and sensitivity to variance in lighting, pose, and expressions.

2)Improved Accuracy (Optimized PCA with Eigenfaces) — ~70%

By computing surrogate covariance (Turk & Pentland method) and selecting the top-k most informative eigenfaces, recognition accuracy increased to ~70%, reflecting better variance retention and dimensionality reduction.

3)Final Accuracy (PCA + ANN Classifier) — ~75%

Integrating PCA feature vectors with an ANN classifier further improved the model's ability to learn complex, nonlinear patterns. This boosted the accuracy to ~75%, along with robust imposter detection, greatly enhancing overall system reliability.

Libraries Used

1.NumPy – matrix manipulation, eigen decomposition
2.SciPy – SVD, linear algebra operations
3.OpenCV (cv2) – reading and preprocessing images
4.Scikit-learn – ANN (MLPClassifier) for classification
5.Matplotlib – plotting accuracy vs k

Project Overview

1.This project performs the complete pipeline needed for PCA-based face recognition:

2.Build the face database

3.Compute the mean face

4.Normalize data (mean-zero)

5.Generate surrogate covariance matrix (Turk & Pentland method)

6.Compute eigenvalues/eigenvectors

7.Select top-k eigenvectors → feature vectors

8.Generate eigenfaces

9.Project images → face signatures

10.Train ANN classifier

11.Test model with:
1)different k values
2)imposters

