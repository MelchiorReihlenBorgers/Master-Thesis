import os
import pandas as pd
import numpy as np

from training.plotting import  plotting_windows
from training.create_training_sample import create_training_sample
from training.extract_features import extract_features

from DataLoad import DataLoad


# Load the .csv file with annotations (obtained from makesense.ai)
if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_annotation.csv"):
    annotations = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

# Compute the mean annotation width and height and subtract the standard deviation once to
# use it as a lower bound in the sampling of random windows in the image.
mean_annotation_width = np.mean(annotations.loc[:,"width"]) - np.std(annotations.loc[:,"width"])
mean_annotation_height = np.mean(annotations.loc[:,"height"]) - np.std(annotations.loc[:,"height"])

# Save the number of rows in the annotations csv file.
# This is used in the subsequent loop to sample 100 random windows per annotation (one annotation per image so far...)
# TODO: Increase K to 10K to make a more reasonable training sample. For now its fine for speed.
nrow = annotations.shape[0]

examples = {}

for row in range(nrow):
    annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[row,"X"], annotations.loc[row,"Y"], annotations.loc[row,"width"], \
                                                                      annotations.loc[row,"height"]

    positive_examples, negative_examples = create_training_sample(annotation_x, annotation_y, annotation_width, annotation_height,
                                                                  K = 100,
                                                                  theta = 0.4,
                                                                  width_low = mean_annotation_width,
                                                                  height_low = mean_annotation_height)

    name = annotations.loc[row,"Image_Name"]

    examples[name] = (positive_examples, negative_examples)


# Plotting
# Plot one example just to show how it works.
x, y, width, height = positive_examples[0][2], positive_examples[0][3], positive_examples[0][0], positive_examples[0][1]
annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[row,"X"], annotations.loc[row,"Y"], annotations.loc[row,"width"], annotations.loc[row,"height"]

plotting_windows("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_image.jpg",
                 x, y, width, height,
                 annotation_x, annotation_y, annotation_width, annotation_height)


# Load the first 3 images to create a training set.
# TODO: This has to be done with at least 50 images. To obtain that you need to annotate the first 50 images as well.
data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")
images, depth, radian = data.load_data(N = 3)


# Extract the features from the samples and save them to create the input data for the algorithm.
# mean_saliency, mean_depth = extract_features()

image_names = list(examples.keys())
all_features = []

# Iterate over different images
for index, name in enumerate(image_names):

    # Each image has positive and negative examples/windows that cover more than theta percent of the annotation and not
    for example in range(2):
        windows_example_list = len(examples[name][example])

        # Extract the features from each of these windows and save them as a list.
        for window in range(windows_example_list):
            mean_depth, mean_saliency = extract_features(depth[index], images[index], examples[name][example][window])

            features = (mean_depth, mean_saliency)

            all_features.append(features)

# TODO: Bind all of these into one data frame.

# TODO: Get target from dict.

