from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def adjusted_rand_index(original_labels, predicted_labels):
    return adjusted_rand_score(original_labels, predicted_labels)


def normalized_mutual_info(original_labels, predicted_labels):
    return adjusted_mutual_info_score(original_labels, predicted_labels)
