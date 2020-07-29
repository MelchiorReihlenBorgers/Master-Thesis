import numpy as np

from training.does_intersect import does_intersect
from training.compute_area import compute_area
from training.compute_overlap import compute_overlap

def create_training_sample(annotation_x, annotation_y, annotation_width, annotation_height, K, width_low, height_low, theta = 0.5):
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