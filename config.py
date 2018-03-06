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

# = Model parameters =
# Clustering parameters
verbose = True # Display extra info?
c = len(features) * 3
m = 2 # Usually set to 2 as starting point
maxiter = 300 # Based on sklearn k-means default
error = 0.0001 # Based on sklearn k-means default
seed = 142 # Seed for randomizer used by clusterer

# Prediction directory paths and file names
pred_dir = "predictions/"
cluster_predictions_path = os.path.join(pred_dir, 'cluster_preds/')
cluster_predictions_filename = 'predictions.txt'

# Model directory paths and file names
save_dir = 'models/'
cluster_centers_path = os.path.join(save_dir, 'cluster_centers/')
cluster_centers_filename = 'centers.txt'

def setup():
	# Create directories required
	# Should only run once at beginning!!!!!!
	directories = [train_processed_path, test_processed_path, cluster_predictions_path, cluster_centers_path, pred_dir, save_dir]
	for directory in directories:
		if not os.path.exists(directory):
			os.makedirs(directory)
