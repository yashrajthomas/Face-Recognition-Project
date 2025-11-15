Implementation of PCA with ANN for Face Recognition

This project implements a Face Recognition System using Principal Component Analysis (PCA) and an Artificial Neural Network (ANN) classifier.
The system follows the classical Eigenfaces + Neural Network approach and is built using Python, NumPy, SciPy, OpenCV, and Scikit-learn (ANN).

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
