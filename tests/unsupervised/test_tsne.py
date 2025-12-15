import numpy as np
import pytest
from ml_algs.unsupervised.tsne import TSNE

# general functionality tests

def test_basic():
    # test basic fit shape and validity
    X = np.random.rand(10, 5)
    tsne = TSNE(n_components=2, n_iter=100)
    tsne.fit(X)
    embedding = tsne.transform()
    
    assert embedding.shape == (10, 2)
    assert not np.any(np.isnan(embedding))

def test_fit_transform_equivalence():
    # test fit_transform = fit + transform (obvious but black box)
    X = np.random.rand(8, 4)
    tsne = TSNE(n_components=3, n_iter=50)
    emb1 = tsne.fit_transform(X)
    tsne2 = TSNE(n_components=3, n_iter=50)
    tsne2.fit(X)
    emb2 = tsne2.transform()
    
    assert emb1.shape == emb2.shape
    assert emb1.shape == (8, 3)

# edge case tests

def test_single_feature():
    # test single feature functionality
    X = np.random.rand(5, 1)
    tsne = TSNE(n_components=2, n_iter=10)
    tsne.fit(X)
    emb = tsne.transform()
    assert emb.shape == (5, 2)
    assert not np.any(np.isnan(emb))

# invalid input tests

def test_empty():
    # test empty input error handling
    X = np.array([]).reshape(0, 0)
    tsne = TSNE()
    with pytest.raises(ValueError):
        tsne.fit(X)

def test_nonnumeric():
    # test nonnumeric input error handling (sorry smiling friends)
    X = np.array([['pim', 'charlie'], ['allan', 'glep']])
    tsne = TSNE()
    with pytest.raises(TypeError):
        tsne.fit(X)

def test_transform_before_fit():
    # test early transform error handling
    tsne = TSNE()
    with pytest.raises(AttributeError):
        tsne.transform()