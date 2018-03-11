import os
import config
import util
import logging
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
from keras.models import Model
from keras.layers import Dense, Input
from keras import losses, metrics

class LDA():
    def __init__(self):
        lda = LinearDiscriminantAnalysis()

    def fit(self, data, labels):
        self.lda.fit(data, labels)

    def predict(self, data):
        return lda.predict_proba(data)

    def get_classes(self):
        return self.lda.classes_

    def save(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        joblib.dump(lda, model_path)


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


class Neural():
    def __init__(self, input_size, num_hidden_neurons, output_size):
        self.input_size = input_size
        self.num_hidden_neurons = num_hidden_neurons
        self.out_size = output_size
        self.model = None

    def build(self, weights_path=None):
        x = Input(shape=(self.input_size, ))
        h = Dense(self.num_hidden_neurons, activation='sigmoid')(x)
        y = Dense(self.out_size, activation='softmax')(h)
        self.model = Model(x, y)
        self.model.compile(optimizer='Adam',
                                    loss=losses.categorical_crossentropy,
                                    metrics=[metrics.categorical_accuracy])
        if weights_path:
            self.model.load_weights(weights_path)

    def save(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.model.save_weights(model_path)

    def fit(self, examples, labels, batch_size, epochs):
        self.model.fit(examples, labels, batch_size=batch_size, epochs=epochs)

    def predict(self, examples):
        predictions = self.model.predict(examples)
        return predictions

    def evaluate(self, examples, labels):
        return self.model.evaluate(examples, labels)


def main():
    # Debugging models and example use
    data_filename = os.path.join(config.train_processed_path, config.examples_filename)
    labels_filename = os.path.join(config.train_processed_path, config.labels_filename)
    hot_labels_filename = os.path.join(config.train_processed_path, config.hot_labels_filename)
    data = pd.read_csv(data_filename, skiprows=0)
    labels = pd.read_csv(labels_filename, skiprows=0)
    hot_labels = pd.read_csv(hot_labels_filename, skiprows=0)
    # example = data.loc[13].reshape((1, data.loc[0].shape[0]))
    # label = labels.loc[13].reshape((1, labels.loc[0].shape[0]))
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
    predictions = predictions.transpose()
    util.save_predictions(predictions, os.path.join(config.cluster_predictions_path, config.cluster_predictions_filename))
    print predictions.shape

    # LDA = LinearDiscriminantAnalysis()
    # LDA.fit(examples, labels)
    # print LDA.classes_
    # predictions = LDA.predict_proba(examples)
    print predictions.shape
    print hot_labels.shape

    NN = Neural(predictions.shape[1], config.num_hidden_neurons, config.output_size)
    NN.build()
    NN.fit(predictions, hot_labels, config.batch_size, config.epochs)



if __name__ == '__main__':
    main()
