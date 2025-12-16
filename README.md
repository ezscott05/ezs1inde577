# Rice University CMOR 438/INDE 577 Final Project
### by: Evelyn Scott

This repository contains Python 3 implementations of common machine learning algorithms written for the final project of the Data Science & Machine Learning course, as well as test suites for these algorithms, all of which are reliably passed.

## Structure

The following list mirrors the structure of both the ml_algs folder, which contains all algorithm implementations, and the tests folder, which contains the corresponding pytest testing suites.

- Supervised
  - Classification
    - KNN (K-Nearest Neighbors) for classification
    - CART (Classification and Regression Tree)
    - Gradient Boosting
    - Perceptron
    - Multilayer Perceptron
    - Random Forest
    - Logistic Regression
  - Regression
    - KNN (K-Nearest Neighbors) for regression
    - Linear Regression
    - Ridge and LASSO Regression
- Unsupervised
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - Hierarchical Clustering (Agglomerative)
  - K-Means Clustering
  - PCA (Principal Component Analysis)
  - t-SNE (t-distributed Stochastic Neighbor Embedding)

Many regression models, such as linear regression, ridge regression, and LASSO are written using gradient descent or coordinate descent instead of the typical normal equation with matrix inversion and such.

Test suites are divided into general functionality tests, which simply use the algorithm for basic tasks and ensures the validity of the results, edge case tests, which test for correct behavior in special cases, such as only a single sample being present, and invalid input tests, which test for proper error raising in the case of the user of the package attempting to perform an invalid action, usually predicting with an unfit model or feeding the model data in the wrong format.

The notebooks folder will contain Jupyter Notebooks demonstrating use of the algorithms once they have been completed.