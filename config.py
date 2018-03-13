import os

# Directory paths and file names
data_dir = 'data/'
train_dir = 'train/'
test_dir = 'test/'
processed_dir = 'processed/'
full_image_dir = 'CBIS-DDSM-FULL/'
roi_image_dir = 'CBIS-DDSM-ROI/'

# Non-processed data file names
train_data_filename = 'mass_case_description_train_set.csv'
test_data_filename = 'mass_case_description_test_set.csv'
# processed data file names
examples_filename = 'examples.csv'
hot_labels_filename = 'hot_labels.csv'
labels_filename = 'labels.csv'
image_examples_filename = 'image_examples.csv'
blob_examples_filename = 'blob_examples.csv'

# Paths to data directories
train_path = os.path.join(data_dir, train_dir)
test_path = os.path.join(data_dir, test_dir)
train_processed_path = os.path.join(train_path, processed_dir)
test_processed_path = os.path.join(test_path, processed_dir)

# Base paths to images
train_full_image_base = os.path.join(train_path, full_image_dir)
train_roi_image_base = os.path.join(train_path, roi_image_dir)
test_full_image_base = os.path.join(test_path, full_image_dir)
test_roi_image_base = os.path.join(test_path, roi_image_dir)

# = Columns to extract =
# List of column name(s) to use as features
features = ['breast_density', 'abnormality id', 'mass shape', 'mass margins', 'subtlety']
# List of column name to use as target (label)
target = ['assessment']

# Image size to apply
# image_size = [5298, 3148] # Set to None to recompute during processing
# image_size = [2649, 1574] # Set to None to recompute during processing
image_size = [1324, 787] # Set to None to recompute during processing
# image_size = [500, 300] # Set to None to recompute during processing

# = Model parameters =
# Clustering parameters
verbose = True # Display extra info?
c = 3
m = 2 # Usually set to 2 as starting point
maxiter = 300 # Based on sklearn k-means default
error = 0.0001 # Based on sklearn k-means default
seed = 142 # Seed for randomizer used by clusterer

# Neural Network parameters
num_hidden_neurons = 10
output_size = 6
batch_size = None
epochs = 50

# Prediction directory paths and file names
pred_dir = "predictions/"
cluster_predictions_path = os.path.join(pred_dir, 'cluster_preds/')
cluster_predictions_filename = 'predictions.txt'
lda_predictions_path = os.path.join(pred_dir, 'lda_preds/')
lda_predictions_filename = 'predictions.txt'
nn_predictions_path = os.path.join(pred_dir, 'neural_net_preds/')
nn_predictions_filename = 'predictions.txt'

# Model directory paths and file names
save_dir = 'models/'
cluster_centers_path = os.path.join(save_dir, 'cluster_centers/')
cluster_centers_filename = 'centers'
lda_model_path = os.path.join(save_dir, 'lda/')
lda_model_filename = 'model.pkl'
nn_model_path = os.path.join(save_dir, 'neural_net/')
nn_model_filename = 'nn.h5'


def setup():
	# Create directories required
	# Should only run once at beginning!!!!!!
	directories = [train_processed_path, test_processed_path, cluster_predictions_path, cluster_centers_path, pred_dir, save_dir]
	for directory in directories:
		if not os.path.exists(directory):
			os.makedirs(directory)
