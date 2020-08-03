import os
import matplotlib.pyplot as plt
import numpy as np

from training.saliency_map_estimation import saliency_map_estimation

def extract_features(depth, image, dict_entry):
    """
    Extract features from the different windows selected.

    1. Compute mean saliency
    2. Compute mean depth

    The intention is that this is to be used in an iterative process where the function is called upon iteratively.

    :param image: Image to be used
    :param dict_entry: Dictionary entry containing (in that order) width, height, x and y coordinate.
    """

    # Make sure that all of the values are integers as otherwhise they are saved as arrays
    # which can not be used to slice arrays.
    width, height, x, y = int(dict_entry[0]), int(dict_entry[1]), int(dict_entry[2]), int(dict_entry[3])

    # Get percentage of coordinates for depth map, i.e. x percent of the length of the image and y percent of the height to
    # map this on the smaller depth map.

    xdim_percentage = x/image.shape[1]
    ydim_percentage = y/image.shape[0]
    width_percentage = width / image.shape[0]
    height_percentage = height/image.shape[1]


    x_depth = int(xdim_percentage * depth.shape[1])
    y_depth = int(ydim_percentage * depth.shape[0])
    width_depth = int(width_percentage * depth.shape[1])
    height_depth = int(height_percentage * depth.shape[0])

    # Get the relevant part of the depth map.
    depth = np.array(depth)
    depth = depth[y_depth:y_depth+height_depth, x_depth:x_depth+width_depth]

    # Resize the image to only compute the saliency on the parts that are relevant
    image = image[y:y + height, x:x + width, :]

    saliencyMap = saliency_map_estimation(image = image)


    # Compute the mean depth and saliency:
    mean_depth = np.mean(depth)
    mean_saliency = np.mean(saliencyMap)

    return mean_depth, mean_saliency

def get_features(label_dictionary, depth, images):
    image_names = list(label_dictionary.keys())
    all_features = []

    # Iterate over different images
    for index, name in enumerate(image_names):

        # Each image has positive and negative examples/windows that cover more than theta percent of the annotation and not
        for example in range(2):
            windows_example_list = len(label_dictionary[name][example])

            # Extract the features from each of these windows and save them as a list.
            for window in range(windows_example_list):
                mean_depth, mean_saliency = extract_features(depth[index], images[index],
                                                             label_dictionary[name][example][window])

                features = (mean_depth, mean_saliency)

                all_features.append(features)

    return all_features






