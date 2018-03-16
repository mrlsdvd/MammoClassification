from __future__ import print_function
import os
import config
import util
import logging
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import shuffle
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


class SoftCluster():
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
        self.centers = dict() # centers map (label -> center)
        self.Us = dict() # Us map (label -> U)

    def fit(self, data, labels):
        # For each label, create fuzzy clusters
        unique_labels = labels[config.target].values.flatten().tolist()
        unique_labels = set(unique_labels)
        unique_labels = sorted(unique_labels) # Sort to keep same order always
        for label in unique_labels:
            print label
            # Get rows of current label
            label_data = data.loc[labels.index[(labels[config.target[0]] == label)].tolist()]
            # fuzz.cluster.cmeans expects data to be of shape (# features, # examples)
            label_data = label_data.transpose()
            # Compute soft clusters for label
            centers, U, _, _, _, p, fpc = fuzz.cluster.cmeans(label_data, c=self.c, m=self.m, error=self.error, maxiter=self.maxiter, init=self.init, seed=self.seed)
            if self.verbose:
                self.logger.info("Took {} iterations".format(p))
                self.logger.info("fpc: {}".format(fpc))
            self.centers[label] = centers
            self.Us[label] = U

    def predict(self, examples):
        # Compute example fit in all soft clusters
        examples = examples.transpose()
        if self.centers is None:
            self.logger.error("Be sure to fit data or set centers, before attempting to predict!")
            return
        Us = []
        for label in sorted(self.centers.keys()):
            centers = self.centers[label]
            U, _, _, _, _, _ = fuzz.cluster.cmeans_predict(examples, cntr_trained=centers, m=self.m, error=self.error, maxiter=self.maxiter, init=self.init, seed=self.seed)
            Us.append(U.T)
        return np.hstack(Us)

    def save_centers(self, filepath):
        if self.centers is None:
            self.logger.error("Be sure to fit data or set centers, before attempting to save!")
            return
        for label in sorted(self.centers.keys()):
            np.savetxt("{}_{}.txt".format(filepath, label), self.centers[label])

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
        if weights_path != None:
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
    images_filename = os.path.join(config.train_processed_path, "image_examples.csv")
    blobs_filename = os.path.join(config.train_processed_path, "blob_examples.csv")
    labels_filename = os.path.join(config.train_processed_path, config.labels_filename)
    hot_labels_filename = os.path.join(config.train_processed_path, config.hot_labels_filename)
    data = pd.read_csv(data_filename, skiprows=0)
    print data.shape
    images = pd.read_csv(images_filename, skiprows=0)
    print images.shape
    blobs = pd.read_csv(blobs_filename, skiprows=0, usecols=range(1, 4))
    print blobs.shape
    labels = pd.read_csv(labels_filename, skiprows=0)
    hot_labels = pd.read_csv(hot_labels_filename, skiprows=0)
    # example = data.loc[13].reshape((1, data.loc[0].shape[0]))
    # label = labels.loc[13].reshape((1, labels.loc[0].shape[0]))
    examples = pd.concat([data, images, blobs], axis=1)

    print examples.shape

    c = 4
    m = 2
    maxiter = 300 # Based on sklearn k-means default
    error = 0.0001 # Based on sklearn k-means default
    seed = 142 # Seed for randomizer used by clusterer

    FCM = SoftCluster(c, m, maxiter, error, init=None, seed=seed)
    FCM.fit(examples, labels)
    FCM.save_centers(os.path.join(config.cluster_centers_path, config.cluster_centers_filename))
    predictions = FCM.predict(examples)
    # predictions = np.log(predictions)
    util.save_predictions(predictions, os.path.join(config.cluster_predictions_path, config.cluster_predictions_filename))
    naiveBayesClass(data, labels)
    print predictions

    # LDA = LinearDiscriminantAnalysis()
    # LDA.fit(examples, labels)
    # print LDA.classes_
    # predictions = LDA.predict_proba(examples)
    print predictions.shape
    print hot_labels.shape

    NN = Neural(predictions.shape[1], config.num_hidden_neurons, config.output_size)
    NN.build()
    shuffled_pred, shuffled_labels = shuffle(predictions, hot_labels.as_matrix(), random_state=142)
    NN.fit(shuffled_pred, shuffled_labels, config.batch_size, config.epochs)

def printStatistics(labels_test_predicted, labels_test_actual):
	numCorrect = 0
	total = 0
	benign = 0
	malignant = 0
	incorrect_matrix = np.zeros((6,6))
	for i in range(len(labels_test_predicted)):
		if labels_test_predicted[i] == labels_test_actual[i]:
			numCorrect += 1
		else:
			incorrect_matrix[labels_test_predicted[i]][labels_test_actual[i]] += 1
		total += 1
	print("Exact prediction accuracy: ")
	print(float(numCorrect)/total)
	print("Incorrect Matrix:")
	print("Actual along top, predicted on side")
	print("\t0\t1\t2\t3\t4\t5")
	for i in range(6):
		print(str(i), end="\t")
		for j in range(6):
			print(str(incorrect_matrix[i][j]), end="\t")
		print()
	numCorrect = 0
	total = 0
	for i in range(len(labels_test_predicted)):
		if labels_test_predicted[i] == labels_test_actual[i]:
			numCorrect +=1
		elif (labels_test_predicted[i] == 4 and labels_test_actual[i] == 5) or (labels_test_predicted[i] == 5 or labels_test_actual[i] == 4):
			numCorrect += 1
		elif (labels_test_predicted[i] == 1 and labels_test_actual[i] == 2) or (labels_test_predicted[i] == 2 or labels_test_actual[i] == 1):
			numCorrect += 1
		total += 1
	print("Accuracy by bucket (0, 1-2, 3, 4-5): ")
	print(float(numCorrect)/total)

def naiveBayesClass(data, labels):
	# train
	images_filename = os.path.join(config.train_processed_path, "image_examples.csv")
	blobs_filename = os.path.join(config.train_processed_path, "blob_examples.csv")
	blobs = pd.read_csv(blobs_filename, skiprows=0)
	examples = pd.concat([blobs, data], axis=1).values.tolist()
	data_train = data.values.tolist()
	labels_train = labels.values.tolist()
	# need list of integers, not list of lists each with a single int
	labels = []
	for label in labels_train:
		labels.append(label[0])
	clf = LogisticRegression()
	clf.fit(examples, labels)
	# test
	data_test_filename = os.path.join(config.test_processed_path, config.examples_filename)
	labels_test_filename = os.path.join(config.test_processed_path, config.labels_filename)	
	data_test = pd.read_csv(data_test_filename, skiprows=0)
	labels_test_actual = pd.read_csv(labels_test_filename, skiprows=0).values.tolist()
	blobs_filename = os.path.join(config.test_processed_path, "blob_examples.csv")
	blobs = pd.read_csv(blobs_filename, skiprows=0)
	examples_test = pd.concat([blobs, data_test], axis=1).values.tolist()
	data_test = data_test.values.tolist()
	labels_test_predicted = clf.predict(examples_test)
	printStatistics(labels_test_predicted, labels_test_actual)

if __name__ == '__main__':
    main()
