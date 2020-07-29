import os
import pandas as pd
import numpy as np

from training.plotting import  plotting_windows

from training.create_training_sample import create_training_sample

if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_annotation.csv"):
    annotations = pd.read_csv("test_annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

mean_annotation_width = np.mean(annotations.loc[:,"width"]) - np.std(annotations.loc[:,"width"])
mean_annotation_height = np.mean(annotations.loc[:,"height"]) - np.std(annotations.loc[:,"height"])


nrow = annotations.shape[0]

examples = {}

for row in range(nrow):
    annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[row,"X"], annotations.loc[row,"Y"], annotations.loc[row,"width"], \
                                                                      annotations.loc[row,"height"]

    positive_examples, negative_examples = create_training_sample(annotation_x, annotation_y, annotation_width, annotation_height,
                                                                  K = 10000,
                                                                  theta = 0.4,
                                                                  width_low = mean_annotation_width,
                                                                  height_low = mean_annotation_height)

    name = annotations.loc[row,"Image_Name"]

    examples[name] = (positive_examples, negative_examples)


# Plotting
x, y, width, height = positive_examples[0][2], positive_examples[0][3], positive_examples[0][0], positive_examples[0][1]
annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[row,"X"], annotations.loc[row,"Y"], annotations.loc[row,"width"], annotations.loc[row,"height"]

plotting_windows("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_image.jpg",
                 x, y, width, height,
                 annotation_x, annotation_y, annotation_width, annotation_height)

