import os
import numpy as np


def save_predictions(predictions, filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savetxt(filepath, predictions)
