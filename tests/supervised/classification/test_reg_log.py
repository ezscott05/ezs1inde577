import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml_algs.supervised.classification.reg_log import Log_Reg

# general functionality tests

def test_simple():
    # and logic
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    
    model = Log_Reg(lr=0.5, epochs=1000)
    model.fit(X, y)
    pred = model.predict(X)
    
    # predictions should match labels
    assert np.array_equal(pred, y)

def test_p_range():
    # test that probabilities fall within the valid range
    X = np.array([[0],[1],[2]])
    y = np.array([0,1,1])
    
    model = Log_Reg(lr=0.1, epochs=500)
    model.fit(X, y)
    probs = model.predict_p(X)
    
    assert np.all(probs >= 0) and np.all(probs <= 1)

def test_iris():
    # test on iris dataset
    iris = load_iris()
    # Only classes 0 and 1
    X = iris.data[iris.target < 2]
    y = iris.target[iris.target < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    model = Log_Reg(lr=0.1, epochs=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    accuracy = np.mean(pred == y_test)
    # Should achieve >90% accuracy on linearly separable subset
    assert accuracy > 0.9

def test_threshold():
    X = np.array([[0],[1]])
    y = np.array([0,1])
    
    model = Log_Reg(lr=0.5, epochs=500)
    model.fit(X, y)
    
    probs = model.predict_p(X)
    pred = model.predict(X)
    
    # predictions should match threshold 0.5
    assert np.all(pred == (probs >= 0.5).astype(int))

# edge case tests

def test_single_sample():
    # test with only one sample to learn from
    X = np.array([[5]])
    y = np.array([1])
    model = Log_Reg(lr=0.1, epochs=500)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred[0] == 1

def test_zero_features():
    # test with only resulting data and no feature data
    X = np.empty((3,0))  # 3 samples, 0 features
    y = np.array([0,1,0])
    model = Log_Reg()
    model.fit(X, y)
    pred = model.predict(X)
    # should predict majority class
    majority = 1  # np.mean(y) >= 0.5 → 1
    assert np.all(pred == majority)

def test_all_labels_identical():
    # test with identical y values
    X = np.array([[1],[2],[3]])
    y = np.array([1,1,1])
    model = Log_Reg()
    model.fit(X, y)
    pred = model.predict(X)
    assert np.all(pred == 1)

# invalid input tests

def test_empty():
    # test proper error handling with empty input
    model = Log_Reg()
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))

def test_mismatch():
    # test proper error handling with a mismatch of X/y sizes
    X = np.array([[1],[2]])
    y = np.array([1])
    model = Log_Reg()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_nonnumeric():
    # test proper error handling with nonnumeric data
    X = np.array([['a'],['b']])
    y = np.array([0,1])
    model = Log_Reg()
    with pytest.raises(TypeError):
        model.fit(X, y)