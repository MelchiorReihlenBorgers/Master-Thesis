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



if __name__ == "__main__":
    from DataLoad import DataLoad

    data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")

    images, depth, radian = data.load_data(N = 1)

    mean_depth, mean_saliency = extract_features(depth[0], images[0], examples['585216074-0.jpg'][0][0])
