import os
import numpy as np
import pandas as pd

def save_predictions(predictions, filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savetxt(filepath, predictions)

def combine_feature_sets(features):
    """
    Combine sets of features for same dataset
    Essentially concatenates fetures for each example
    features: list of features for same examples (list of pandas dataframes)
    """
    return pd.concat(features, axis=1)

def convert_index_to_label(indices, label_map):
    """
    Given indices=[0, 1, 2] and label_map={0: '1', 1: '2', 2: '3'},
    returns [1, 2, 3]
    """
    labels = []
    for i in indices:
        labels.append(label_map[i])
    return labels
