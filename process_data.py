import sys
import os
import config
import util
import numpy as np
import pandas as pd

def main(dataset):
    if dataset == 'train':
        dataset_filename = os.path.join(config.train_path, config.train_data_filename)
        processed_examples_filename = os.path.join(config.train_processed_path, config.examples_filename)
        processed_hot_labels_filename = os.path.join(config.train_processed_path, config.hot_labels_filename)
        processed_labels_filename = os.path.join(config.train_processed_path, config.labels_filename)
    else:
        # Use test set
        dataset_filename = os.path.join(config.test_path, config.test_data_filename)
        processed_examples_filename = os.path.join(config.test_processed_path, config.examples_filename)
        processed_hot_labels_filename = os.path.join(config.test_processed_path, config.hot_labels_filename)
        processed_labels_filename = os.path.join(config.test_processed_path, config.labels_filename)

    features = config.features
    target = config.target

    data = pd.read_csv(dataset_filename)
    # Separate features and labels
    examples = data[features]
    labels = data[target]
    # One hot encode example features
    one_hot_encoded_examples = pd.get_dummies(examples, prefix=features, columns=features)
    # one hot encode label values
    one_hot_encoded_labels = pd.get_dummies(labels, prefix=target, columns=target)

    # Save examples and labels to file
    one_hot_encoded_examples.to_csv(processed_examples_filename, header=True, index=False)
    one_hot_encoded_labels.to_csv(processed_hot_labels_filename, header=True, index=False)
    labels.to_csv(processed_labels_filename, header=True, index=False)


if __name__ == '__main__':
    dataset = sys.argv[1] # 'test' or 'train'
    main(dataset)
