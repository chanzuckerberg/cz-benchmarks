import numpy as np
from czibench.metrics.embedding import silhouette_score


def test_silhouette_score():
    embedding = np.random.rand(100, 10)
    labels = np.random.randint(0, 5, 100)
    score = silhouette_score(embedding, labels)
    assert isinstance(score, float), "Silhouette score should return a float value"
