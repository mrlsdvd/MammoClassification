import sys
import os
import config
import util
import numpy as np
import pandas as pd

def main():
    train_dataset_filename = os.path.join(config.train_path, config.train_data_filename)
    train_processed_examples_filename = os.path.join(config.train_processed_path, config.examples_filename)
    train_processed_hot_labels_filename = os.path.join(config.train_processed_path, config.hot_labels_filename)
    train_processed_labels_filename = os.path.join(config.train_processed_path, config.labels_filename) 
    test_dataset_filename = os.path.join(config.test_path, config.test_data_filename)
    test_processed_examples_filename = os.path.join(config.test_processed_path, config.examples_filename)
    test_processed_hot_labels_filename = os.path.join(config.test_processed_path, config.hot_labels_filename)
    test_processed_labels_filename = os.path.join(config.test_processed_path, config.labels_filename)

    features = config.features
    target = config.target

    train_data = pd.read_csv(train_dataset_filename)
    test_data = pd.read_csv(test_dataset_filename)
    # Separate features and labels
    train_examples = train_data[features]
    train_labels = train_data[target]
    test_examples = test_data[features]
    test_labels = test_data[target]
    # Stack train and test (to create consistent examples and labels)
    train_examples_shape = train_examples.shape
    test_examples_shape = test_examples.shape
    train_labels_shape = train_labels.shape
    test_labels_shape = test_labels.shape

    combined_examples = pd.concat([train_examples, test_examples], axis=0) # Remove headers from test
    combined_labels = pd.concat([train_labels, test_labels], axis=0) # Remove headers from test

    # One hot encode example features
    one_hot_encoded_examples = pd.get_dummies(combined_examples, prefix=features, columns=features)
    # one hot encode label values
    one_hot_encoded_labels = pd.get_dummies(combined_labels, prefix=target, columns=target)

    # Separate train and test set examples and labels
    one_hot_encoded_train_examples = one_hot_encoded_examples.iloc[:train_examples_shape[0]]
    one_hot_encoded_test_examples = one_hot_encoded_examples.iloc[train_examples_shape[0]:]

    one_hot_encoded_train_labels = one_hot_encoded_labels.iloc[:train_labels_shape[0]]
    one_hot_encoded_test_labels = one_hot_encoded_labels.iloc[train_labels_shape[0]:]

    # Insert headers back into test examples and labels
    # one_hot_encoded_test_examples = pd.concat([one_hot_encoded_train_examples, one_hot_encoded_test_examples])
    # one_hot_encoded_test_labels = pd.concat([one_hot_encoded_train_labels, one_hot_encoded_test_labels])

    # Save examples and labels to file
    one_hot_encoded_train_examples.to_csv(train_processed_examples_filename, header=True, index=False)
    one_hot_encoded_train_labels.to_csv(train_processed_hot_labels_filename, header=True, index=False)
    train_labels.to_csv(train_processed_labels_filename, header=True, index=False)

    one_hot_encoded_test_examples.to_csv(test_processed_examples_filename, header=True, index=False)
    one_hot_encoded_test_labels.to_csv(test_processed_hot_labels_filename, header=True, index=False)
    test_labels.to_csv(test_processed_labels_filename, header=True, index=False)


if __name__ == '__main__':
    main()
