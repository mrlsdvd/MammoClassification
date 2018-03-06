import os
import config
import util
import logging
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class FuzzyCMEans():
    def __init__(self, c, m, maxiter, error, init=None, seed=None, verbose=True):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.c = c
        self.m = m
        self.maxiter = maxiter
        self.error = error
        self.init = init
        self.seed = seed
        self.centers = None
        self.U = None

    def fit(self, data):
        # fuzz.cluster.cmeans expects data to be of shape (# features, # examples)
        data = data.transpose()
        centers, U, _, _, _, p, fpc = fuzz.cluster.cmeans(data, c=self.c, m=self.m, error=self.error, maxiter=self.maxiter, init=self.init, seed=self.seed)
        if self.verbose:
            self.logger.info("Took {} iterations".format(p))
            self.logger.info("fpc: {}".format(fpc))
        self.centers = centers
        self.U = U

    def predict(self, examples):
        examples = examples.transpose()
        if self.centers is None:
            self.logger.error("Be sure to fit data or set centers, before attempting to predict!")
            return
        U, _, _, _, _, _ = fuzz.cluster.cmeans_predict(examples, cntr_trained=self.centers, m=self.m, error=self.error, maxiter=self.maxiter, init=self.init, seed=self.seed)
        return U

    def save_centers(self, filepath):
        if self.centers is None:
            self.logger.error("Be sure to fit data or set centers, before attempting to save!")
            return
        np.savetxt(filepath, self.centers)

    def set_centers(self, centers):
        self.centers = centers


def main():
    # Debugging models and example use
    data_filename = os.path.join(config.train_processed_path, config.examples_filename)
    labels_filename = os.path.join(config.train_processed_path, config.labels_filename)
    data = pd.read_csv(data_filename, skiprows=0)
    labels = pd.read_csv(labels_filename, skiprows=0)
    example = data.loc[5].reshape((1, data.loc[0].shape[0]))
    label = labels.loc[5].reshape((1, labels.loc[0].shape[0]))
    examples = data

    c = 10
    m = 5
    maxiter = 300 # Based on sklearn k-means default
    error = 0.0001 # Based on sklearn k-means default
    seed = 142 # Seed for randomizer used by clusterer

    FCM = FuzzyCMEans(c, m, maxiter, error, init=None, seed=seed)
    FCM.fit(data)
    FCM.save_centers(os.path.join(config.cluster_centers_path, config.cluster_centers_filename))
    predictions = FCM.predict(examples)
    util.save_predictions(predictions, os.path.join(config.cluster_predictions_path, config.cluster_predictions_filename))

    # QDA = QuadraticDiscriminantAnalysis()
    # QDA.fit(examples, labels)
    # print QDA.predict_proba(example)


if __name__ == '__main__':
    main()
