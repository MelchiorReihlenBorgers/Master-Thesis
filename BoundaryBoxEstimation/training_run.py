import os
import numpy as np
import pandas as pd

from training.create_training_sample import create_training_sample
from training.extract_features import get_features

from training.extract_features import extract_features

from DataLoad import DataLoad


# Load the .csv file with annotations (obtained from makesense.ai)
if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv"):
    annotations = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

# Compute the mean annotation width and height and subtract the standard deviation once to
# use it as a lower bound in the sampling of random windows in the image.
mean_annotation_width = np.mean(annotations.loc[:,"width"]) - np.std(annotations.loc[:,"width"])
mean_annotation_height = np.mean(annotations.loc[:,"height"]) - np.std(annotations.loc[:,"height"])

# Save the number of rows in the annotations csv file.
# This is used in the subsequent loop to sample 100 random windows per annotation (one annotation per image so far...)
label_dictionary, labels,  IoUs = create_training_sample(annotations = annotations,
                                                         K = 100, theta = 0.6, width_low = mean_annotation_width,
                                                         height_low = mean_annotation_height)
# Load all images
data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")
N = len(data.extract_paths()[0])
images, depth, radian = data.load_data(N = N)


# Extract the features from the samples and save them to create the input data for the algorithm.
# mean_saliency, mean_depth = extract_features()


image_names = list(label_dictionary.keys())

all_features = []

# Extract all features from the images.
for index, name in enumerate(image_names):
    print("{:.2f}% of images processed.".format(100*((index+1)/len(image_names))))

    for label in range(2):
        windows_example_list = len(label_dictionary[name][label])
        print("This is the length of the positive and negative examples: {}".format(windows_example_list))

        for window in range(windows_example_list):
            mean_depth, mean_saliency, color_contrast, edge_density, sd_depth, sd_saliency, depth_contrast, binary_saliency = extract_features(depth[ index ],
                                                                                                              images[ index ],
                                                                                                              label_dictionary[name][label][window],
                                                                                                              beta=0.1)

            features = (mean_depth, mean_saliency, color_contrast, edge_density, sd_depth, sd_saliency, depth_contrast, binary_saliency)

            all_features.append(features)


# Create one data frame with all features
col_names = ["Mean_Depth", "Mean_Saliency", "Color_Contrast", "Edge_Density", "SD_Depth", "SD_Saliency",
             "Depth_Contrast", "Binary_Saliency"]

data = pd.DataFrame(all_features, columns = col_names)
data["label"] = labels

data.to_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/data.csv")


