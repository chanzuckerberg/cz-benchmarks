from sklearn.metrics import silhouette_score as sklearn_silhouette_score


def silhouette_score(embedding, labels):
    return sklearn_silhouette_score(embedding, labels)
