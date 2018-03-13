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
    naiveBayesClass(data, labels)
    # QDA = QuadraticDiscriminantAnalysis()
    # QDA.fit(examples, labels)
    # print QDA.predict_proba(example)

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
