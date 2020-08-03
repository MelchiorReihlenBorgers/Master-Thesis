import numpy as np

from itertools import chain

from training.does_intersect import does_intersect
from training.compute_area import compute_area
from training.compute_overlap import compute_overlap

def label_data(annotation_x, annotation_y, annotation_width, annotation_height, K, width_low, height_low, theta = 0.5):
    """
    Function to create training sample.

    :param K: Number of windows simulated per image.
    :param theta: Threshold for positive image classification
    :return:
    """
    # Create some space to safe the positive and negative examples (i.e. windows that overlap enough and windows that do not)
    positive_examples = []
    negative_examples = []


    for _ in range(K):
        intersection, width, height, x, y  = does_intersect(annotation_x, annotation_y, annotation_width, annotation_height, width_low = width_low, height_low= height_low)

        # If they overlap, compute by how much they overlap
        if intersection:
            area_annotation = compute_area(width = annotation_width, height = annotation_height)
            area_window = compute_area(width = width, height = height)

            overlap = compute_overlap(x,y,width, height,
                                      annotation_x, annotation_y, annotation_width, annotation_height)

            # Is the overlap greater that 0.5 of the size of the sum of the annotation window and the simulated window?
            overlap_criterion = overlap/(area_annotation + area_window) > theta

            if overlap_criterion:
                positive_examples.append(np.array([width, height, x, y]))

            else:
                negative_examples.append(np.array([width, height, x, y]))

    return positive_examples, negative_examples

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

    nrow = annotations.shape[0]

    for row in range(nrow):
        annotation_x, annotation_y, annotation_width, annotation_height = \
            annotations.loc[ row, "X" ], annotations.loc[row, "Y" ], annotations.loc[ row, "width" ], \
            annotations.loc[ row, "height" ]

        positive_examples, negative_examples = label_data(annotation_x, annotation_y, annotation_width,
                                                          annotation_height,
                                                          K=K,
                                                          theta=theta,
                                                          width_low=width_low,
                                                          height_low=height_low)

        N_positives, N_negatives = len(positive_examples), len(negative_examples)

        label = [1] * N_positives + [0] * N_negatives

        labels.append(label)

        name = annotations.loc[ row, "Image_Name" ]

        examples[ name ] = (positive_examples, negative_examples)

    labels = list(chain.from_iterable(labels))

    return examples, labels, positive_examples, negative_examples
