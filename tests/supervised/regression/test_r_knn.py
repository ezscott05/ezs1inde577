import numpy as np
import pytest
from ml_algs.supervised.regression.r_knn import KNN_Reg
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# general functionality tests

def test_reg_known():
    # small regression test
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = KNN_Reg(k=2)
    model.fit(X, y)

    preds = model.predict(np.array([[1.5]]))
    # 1.5 is nearest (1,1) and (2,2) should avg to 1.5
    assert np.isclose(preds[0], 1.5, atol=1e-6)


def test_reg_large():
    # diabetes decent set for regression
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=123)

    model = KNN_Reg(k=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    # loose check to make sure it generally works
    assert mse < 10000

# edge case tests

def test_k_too_large():
    # test choosing more neighbors than there are points
    X_train = np.array([[0], [1]])
    y_train = np.array([0.0, 1.0])

    model = KNN_Reg(k=5)
    model.fit(X_train, y_train)
    pred = model.predict(np.array([[0.5]]))

    # prediction should be in range
    assert 0.0 <= pred[0] <= 1.0

def test_equidistant():
    # test with equidistant neighbors
    X_train = np.array([[0], [2]])
    y_train = np.array([1.0, 3.0])
    model = KNN_Reg(k=2)
    model.fit(X_train, y_train)

    # test point is equidistant from both training points
    pred = model.predict(np.array([[1]]))
    # should average the two :p
    assert np.isclose(pred[0], 2.0)

# invalid input tests

def test_empty():
    # test proper error handling with empty input
    model = KNN_Reg(k=1)
    model.fit(np.array([[0]]), np.array([0]))

    with pytest.raises(ValueError):
        model.predict(np.array([]))

def test_nonnumeric():
    # test proper error handling with nonnumeric data
    model = KNN_Reg(k=1)
    model.fit(np.array([[0]]), np.array([0]))

    with pytest.raises(TypeError):
        model.predict([["a"]])