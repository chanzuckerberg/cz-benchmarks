import numpy as np
from czibench.metrics.clustering import adjusted_rand_index, normalized_mutual_info


def test_adjusted_rand_index():
    original_labels = np.array([0, 0, 1, 1, 2, 2])
    predicted_labels = np.array([1, 1, 0, 0, 2, 2])
    ari = adjusted_rand_index(original_labels, predicted_labels)
    assert isinstance(ari, float), "ARI should return a float value"
    assert 0.0 <= ari <= 1.0, "ARI should be between 0 and 1"


def test_adjusted_rand_index_identical():
    original_labels = np.array([0, 1, 2, 3, 4])
    predicted_labels = np.array([0, 1, 2, 3, 4])
    ari = adjusted_rand_index(original_labels, predicted_labels)
    assert ari == 1.0, "ARI should be 1 for identical labels"


def test_adjusted_rand_index_random():
    original_labels = np.random.randint(0, 5, 100)
    predicted_labels = np.random.randint(0, 5, 100)
    ari = adjusted_rand_index(original_labels, predicted_labels)
    assert -1.0 <= ari <= 1.0, "ARI should be between -1 and 1 even for random labels"


def test_normalized_mutual_info():
    original_labels = np.array([0, 0, 1, 1, 2, 2])
    predicted_labels = np.array([1, 1, 0, 0, 2, 2])
    nmi = normalized_mutual_info(original_labels, predicted_labels)
    assert isinstance(nmi, float), "NMI should return a float value"
    assert 0.0 <= nmi <= 1.0, "NMI should be between 0 and 1"


def test_normalized_mutual_info_identical():
    original_labels = np.array([0, 1, 2, 3, 4])
    predicted_labels = np.array([0, 1, 2, 3, 4])
    nmi = normalized_mutual_info(original_labels, predicted_labels)
    assert nmi == 1.0, "NMI should be 1 for identical labels"
