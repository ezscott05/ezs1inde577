import numpy as np
import pytest
from ml_algs.supervised.regression.reg_lin import Lin_Reg
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# general functionality tests

def test_linear_regression_synthetic():
    # linear data
    X = np.array([[0], [1], [2], [3]])
    y = np.array([1, 3, 5, 7])

    model = Lin_Reg(lr=0.1, epochs=500)
    model.fit(X, y)

    preds = model.predict(np.array([[4], [5]]))
    assert np.allclose(preds, [9, 11], atol=0.1)

def test_linear_regression_real_data():
    # diabetes dataset
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=124
    )

    model = Lin_Reg(lr=0.015, epochs=2500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    # mse check
    assert mse < 5000

# edge case tests

def test_single_sample():
    X = np.array([[5]])
    y = np.array([10])
    model = Lin_Reg(lr=0.05, epochs=1000)
    model.fit(X, y)
    pred = model.predict(X)
    assert np.isclose(pred[0], 10, atol=0.1)

def test_zero_features():
    X = np.empty((3, 0))
    y = np.array([1, 2, 3])
    model = Lin_Reg()
    model.fit(X, y)
    preds = model.predict(X)
    # should predict mean with no feature data
    assert np.allclose(preds, np.mean(y))

def test_constant_feature():
    X = np.array([[1], [1], [1]])
    y = np.array([2, 2, 2])
    model = Lin_Reg(lr=0.05, epochs=1000)
    model.fit(X, y)
    preds = model.predict(X)
    # should pred the constant value
    assert np.allclose(preds, y, atol=0.1)

# invalid input tests

def test_empty():
    model = Lin_Reg()
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))

def test_mismatch():
    X = np.array([[1], [2]])
    y = np.array([1])
    model = Lin_Reg()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_nonnumeric():
    X = np.array([['a'],['b']])
    y = np.array([0,1])
    model = Lin_Reg()
    with pytest.raises(TypeError):
        model.fit(X, y)