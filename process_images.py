import sys
import os
import config
import util
import numpy as np
import pandas as pd
import pydicom as pdcm
from skimage.transform import resize
from skimage.feature import blob_dog, greycomatrix, greycoprops
from sklearn.decomposition import PCA

def flat(image):
    """
    Simply flatten image
    """
    return image.flatten()

def detect_blob(image):
    """
    Use skimage.feature.blob_dog to find blobs in image
    Return np.array(# blobs found, sigma for largest blob, sigma for smallest blob)
    Note: radius of blob is approximately sqrt(2)*sigma
    """
    blobs = blob_dog(image)
    num_blobs = blobs.shape[0]
    max_blob_sigma = np.max(blobs[:, 2])
    min_blob_sigma = np.min(blobs[:, 2])
    return np.array([num_blobs, max_blob_sigma, min_blob_sigma])

def SGLDM(image, props=['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']):
    """
    Use skimage.feature.greycomatrix and skimage.feature.greycoprops
    Return np.array(prop values)
    Note: greyprops makes use of output of greycomatrix
    """
    # distance and angle params for greycomatrix (defined in LNCS )
    # https://link.springer.com/content/pdf/10.1007/11492542_61.pdf
    distances = range(1, 5)
    angles = np.array([0, .25, .5, .75]) * np.pi
    comatrix = greycomatrix(image, distances=distances, angles=angles, levels=np.max(image) + 1)
    properties = []
    for prop in props:
        feature = greycoprops(comatrix, prop=prop)
        properties.append(feature)
    return properties.flatten()


def reduce_dimensionality(features, out_size=50):
    """
    Use PCA to reduce the dimensionality of the examples
    Return reduced examples with size out_size
    """
    pca = PCA(n_components=out_size)
    return pca.fit_transform(features)


def main(dataset, feature_extractors, dim_reduce=False):
    # Define inputs and outputs paths
    if dataset == 'train':
        image_path_base = config.train_full_image_base
        dataset_filename = os.path.join(config.train_path, config.train_data_filename)
        image_examples_filename = os.path.join(config.train_processed_path, config.image_examples_filename)
    else:
        image_path_base = config.test_full_image_base
        dataset_filename = os.path.join(config.test_path, config.test_data_filename)
        image_examples_filename = os.path.join(config.test_processed_path, config.image_examples_filename)
    # Load base description csv
    data = pd.read_csv(dataset_filename)

    # Determine average image size if necessary
    image_size = config.image_size
    if image_size == None:
        shapes = []
        print "Computing average image size"
        for index, row in data.iterrows():
            image_path = os.path.join(image_path_base, row['image file path'])
            dcm = pdcm.filereader.dcmread(image_path)
            image = dcm.pixel_array
            shapes.append(np.array(image.shape))
        image_size = np.mean(shapes, axis=0).astype(int).tolist()
        print "Using size: {}".format(average_shape)

    # For each image path, get image and call feature extractors on image
    combined_example_features = None
    for index, row in data.iterrows():
        image_path = os.path.join(image_path_base, row['image file path'])
        print "Processing: {}".format(index)
        dcm = pdcm.filereader.dcmread(image_path)
        image = dcm.pixel_array
        # Resize image
        image = resize(image, output_shape=image_size)
        extracted_features = []
        for feature_extractor in feature_extractors:
            features = feature_extractor(image)
            extracted_features.append(features)
        # Combine features
        combined_features = np.hstack(extracted_features)
        if combined_example_features is None:
            combined_example_features = combined_features
        else:
            combined_example_features = np.vstack([combined_example_features, combined_features])

    # Reduce dimensionality if necessary
    if dim_reduce:
        combined_example_features = reduce_dimensionality(combined_example_features, 20)
    # Save combined features
    print "Saving {} x {} feature matrix".format(combined_example_features.shape)
    feature_df = pd.DataFrame(combined_example_features)
    feature_df.to_csv(image_examples_filename)


if __name__ == '__main__':
    dataset = sys.argv[1] # train or test
    feature_extractors = [flat]
    main(dataset, feature_extractors, dim_reduce=True)
