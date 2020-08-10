import numpy as np

from itertools import chain

from training.does_intersect import does_intersect
from training.compute_area import compute_area
from training.compute_overlap import compute_overlap

def label_data(annotation_x, annotation_y, annotation_width, annotation_height, K, width_low, height_low, theta):
    """
    Function to create training sample.

    :param K: Number of windows simulated per image.
    :param theta: Threshold for positive image classification
    :return:
    """
    # Create some space to safe the positive and negative examples (i.e. windows that overlap enough and windows that do not)
    positive_examples = []
    negative_examples = []
    IoUs = []


    for _ in range(K):
        intersection, width, height, x, y  = does_intersect(annotation_x, annotation_y, annotation_width, annotation_height, width_low = width_low, height_low= height_low)

        # If they overlap, compute by how much they overlap
        if intersection:
           area_annotation = compute_area(width = annotation_width, height = annotation_height)
           area_window = compute_area(width = width, height = height)

           overlap = compute_overlap(x,y,width, height,
                                      annotation_x, annotation_y, annotation_width, annotation_height)

           # Calculate the Intersection over Union
           IoU = overlap / (area_annotation + area_window)
           IoUs.append(IoU)


           # If the IoU is greater than theta, it is a positive example else negative
           if IoU > theta:
               positive_examples.append(np.array([width, height, x, y]))


           else:
               negative_examples.append(np.array([width, height, x, y]))


        else:
           IoU = 0
           IoUs.append(IoU)

    return positive_examples, negative_examples, IoUs





def create_training_sample(annotations, K, width_low, height_low, theta = 0.4):
    """
    Function to create a dictionary with key = Image Name and value = Tuple (feature1, feature2,...) and a list of labels.

    :param annotations: Annotations data set for rectangles, obtained from makesense.ai. Column names have to be
                        ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"]
    :param K: Number of random windows simulated per image.
    :param width_low: Lowest value for width of simulation (set to mean_width_annotation - sd_width_annotation
    :param height_low: Lowest value for height of simulation (set to mean_height_annotation - sd_height_annotation
    :param theta: Threshold for assigning the label. Default is 0.4 indicating that the simulated window
                  and the union of the annotation and simulated window have to be greater equal 40%.

    :return:
    """

    examples = {}
    labels = []
    IoUs = []

    nrow = annotations.shape[0]

    for row in range(nrow):
        annotation_x, annotation_y, annotation_width, annotation_height = \
            annotations.loc[ row, "X" ], annotations.loc[row, "Y" ], annotations.loc[ row, "width" ], \
            annotations.loc[ row, "height" ]

        positive_examples, negative_examples, IoU = label_data(annotation_x, annotation_y, annotation_width,
                                                          annotation_height,
                                                          K=K,
                                                          theta=theta,
                                                          width_low=width_low,
                                                          height_low=height_low)

        # Get the size of the number of IoUs > theta and smaller
        N_positives, N_negatives = len(positive_examples), len(negative_examples)

        # Create a list with length N_positives + N_Negatives, containing a one or a zero, without using a loop.
        label = [1] * N_positives + [0] * N_negatives

        labels.append(label)


        name = annotations.loc[ row, "Image_Name" ]

        examples[ name ] = (positive_examples, negative_examples)

        IoUs.append(IoU)

    labels = list(chain.from_iterable(labels))
    IoUs = list(chain.from_iterable(IoUs))

    return examples, labels, IoUs


"""
import os
import numpy as np
import pandas as pd


# Load the .csv file with annotations (obtained from makesense.ai)
if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv"):
    annotations = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

# Compute the mean annotation width and height and subtract the standard deviation once to
# use it as a lower bound in the sampling of random windows in the image.
mean_annotation_width = np.mean(annotations.loc[:,"width"]) - np.std(annotations.loc[:,"width"])
mean_annotation_height = np.mean(annotations.loc[:,"height"]) - np.std(annotations.loc[:,"height"])

annotation_x, annotation_y, annotation_width, annotation_height = \
            annotations.loc[0, "X" ], annotations.loc[0, "Y" ], annotations.loc[0, "width" ], \
            annotations.loc[0, "height" ]

K = 100

a,b, c = label_data(annotation_x, annotation_y, annotation_width, annotation_height, K, mean_annotation_width, mean_annotation_height, 0.3)

annotation = annotations.loc[:2,:]

examples, labels, positive_examples, negative_examples, IoUs = \
    create_training_sample(annotations = annotation, K = K, width_low= mean_annotation_width, height_low= mean_annotation_height, theta = 0.3)
"""

