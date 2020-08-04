import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from training.saliency_map_estimation import saliency_map_estimation
from training.expand_image import expand_image

def canny_detector(image, sigma = 0.33):
    """
    Function to do automatic canny edge detection. Code inspired by
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    :param image: Input image
    :param sigma: Deviation from median.
    :return: Binary edge map
    """

    median = np.median(image)
    upper = int(max(0,median*(1-sigma)))
    lower = int(min(255, median*(1+sigma)))

    edged = cv2.Canny(image, lower, upper)

    return edged

def extract_features(depth, image, dict_entry, beta=0.1):
    """
    Extract features from the different windows selected.

    1. Compute mean saliency
    2. Compute mean depth
    3. Compute the Chi-Square color distance between the expanded cropped image and the cropped image
    4. Compute the edge density of the shrunken cropped image

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

    # Expand the image (for Color Contras Feature)
    image_expanded = expand_image(originial_img = image, x = x, y = y, width = width, height = height, beta =beta)

    # Shrink the image (for Edge Density Feature)
    image_shrunken = expand_image(originial_img = image, x = x, y = y, width = width, height = height, beta =-beta)


    # Resize the image to only compute the saliency on the parts that are relevant
    image = image[y:y + height, x:x + width, :]

    saliencyMap = saliency_map_estimation(image = image)


    # Compute Features:

    ## 1) Mean Depth
    mean_depth = np.mean(depth)

    ## 2) Mean Saliency
    mean_saliency = np.mean(saliencyMap)

    ## 3) Color Distance.

    # Compute histogram for each color channel for the cropped image and the expansion of the cropped image.
    hists_cropped = cv2.calcHist(images = image, channels = [0,1,2], mask = None, histSize = [8,8,8], ranges = [0, 256, 0, 256, 0, 256])
    hists_expanded = cv2.calcHist(images = image_expanded, channels = [0,1,2], mask = None, histSize = [8,8,8], ranges = [0, 256, 0, 256, 0, 256])

    # Compute the distance between the histograms using the chi-squared method.
    distance_histograms = cv2.compareHist(hists_cropped, hists_expanded, cv2.HISTCMP_CHISQR)

    ## 4) Edge Density
    edges = canny_detector(image = image, sigma = 0.33)
    edge_density = np.sum(edges)/(edges.shape[0]*edges.shape[1])

    return mean_depth, mean_saliency, distance_histograms, edge_density



def get_features(label_dictionary, depth, images, beta):
    image_names = list(label_dictionary.keys())
    all_features = []

    # Iterate over different images
    for index, name in enumerate(image_names):

        # Each image has positive and negative examples/windows that cover more than theta percent of the annotation and not
        for example in range(2):
            windows_example_list = len(label_dictionary[name][example])

            # Extract the features from each of these windows and save them as a list.
            for window in range(windows_example_list):
                mean_depth, mean_saliency, color_distance, edge_density = extract_features(depth[index], images[index],
                                                                                           label_dictionary[name][example][window],
                                                                                           beta = 0.1)

                features = (mean_depth, mean_saliency, color_distance, edge_density)

                all_features.append(features)

    return all_features






