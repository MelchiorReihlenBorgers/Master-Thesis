import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from training.saliency_map_estimation import saliency_map_estimation
from training.expand_image import expand_image
from training.compute_area import compute_area

from training.euc_distance import euc_distance

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

min_vectorized = np.vectorize(min)


def superpixel_straddeling(image, x, y, width, height):
    """
    :param image: (Full, i.e. non-cropped) input image
    :param x: X coordinate of simulated window
    :param y: Y coordinate of simulated window
    :param width: Width of simulated window
    :param height: Height of simulated window
    :return: Superpixel Straddel
    """
    superpixels = slic(image, 3)

    # Compute the sizes of the superpixels in the entire image
    unique_superpixels = np.unique(superpixels)
    size_superpixels_total = {i:np.count_nonzero(superpixels == unique_superpixels[i]) for i in range(len(unique_superpixels))}

    # Compute size superpixel inside window/cropped image
    superpixels_cropped = superpixels[y:y+height, x:x+width]
    unique_superpixels_inside_window = np.unique(superpixels_cropped)

    ####### Create Pixel Count Inside Window #######
    # If the superpixel is inside the window, | count the occurrences | the occurrences outside are the diff.
    #   between the inside and total

    # Else | set the count to zero | set the count of occurrences outside equal to the total number of pixels
    size_superpixels_inside_window = [None]*len(unique_superpixels)
    size_superpixels_outside_window = [None]*len(unique_superpixels)

    for index, superpixel in enumerate(unique_superpixels):
        if superpixel in unique_superpixels_inside_window:

            size_superpixels_inside_window[index] = np.count_nonzero(superpixels_cropped == superpixel)

            size_superpixels_outside_window[index] = size_superpixels_total[superpixel] - size_superpixels_inside_window[index]

        else:

            size_superpixels_inside_window[index] = 0

            size_superpixels_outside_window[index] = size_superpixels_total[superpixel]

    # Compute the area of the simulated window
    area_window = compute_area(width, height)

    # Compute the contribution
    contribution = min_vectorized(size_superpixels_outside_window, size_superpixels_inside_window)/area_window

    # Use equation (4) from "What is an object?" to obtain the SS measure.
    superpixel_straddel = 1 - np.sum(contribution)

    return superpixel_straddel


def extract_features(depth, image, dict_entry, beta=0.1):
    """
    Extract features from the different windows selected.

    1. Compute mean saliency
    2. Compute mean depth
    3. Compute the Chi-Square color distance between the expanded cropped image and the cropped image
    4. Compute the edge density of the shrunken cropped image
    5. Superpixel Stradelling

    The intention is that this is to be used in an iterative process where the function is called upon iteratively.

    :param image: Image to be used
    :param depth: Depth map corresponding to the image
    :param dict_entry: Dictionary entry containing (in that order) width, height, x and y coordinate of the
                       simulated window.
    :param beta: Hyperparameter controlling by how much the windows should be resized for color contrast and
                 edge density feature.

    :return mean_depth, mean_saliency, distance histograms, edge_density are the parameters described above.

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

    # Expanded height and width for depth map used for Depth Contrast
    x_depth_new = int(max(0, x_depth - 0.5*beta*width))
    y_depth_new = int(max(0, y_depth - 0.5*beta*height))

    width_depth_new = int(width_depth * (1 + beta))
    height_depth_new = int(height_depth * (1 + beta))

    if width_depth_new + x_depth_new > depth.shape[1]:
        width_depth_new = depth.shape[1] - x_depth_new

    if height_depth_new + y_depth_new > depth.shape[0]:
        height_depth_new = depth.shape[0] - y_depth_new


    # Get the relevant part of the depth map.
    depth = np.array(depth)
    depth_cropped = depth[y_depth:y_depth+height_depth, x_depth:x_depth+width_depth]

    # Expand the depth map.
    depth_expanded = depth[y_depth_new:y_depth_new+height_depth_new, x_depth_new:x_depth_new+width_depth_new]

    # Expand the image (for Color Contras Feature)
    image_expanded = expand_image(originial_img = image, x = x, y = y, width = width, height = height, beta =beta)

    # Shrink the image (for Edge Density Feature)
    image_shrunken = expand_image(originial_img = image, x = x, y = y, width = width, height = height, beta =-beta)


    # Resize the image to only compute the saliency on the parts that are relevant
    image_cropped = image[y:y + height, x:x + width, :]

    saliencyMap = saliency_map_estimation(image = image_cropped)


    # Compute Features:

    ## 1) Mean Depth
    mean_depth = np.mean(depth_cropped)

    ## 2) Mean Saliency
    mean_saliency = np.mean(saliencyMap)

    ## 3) Color Distance.

    # Compute histogram for each color channel for the cropped image and the expansion of the cropped image.
    hists_cropped = cv2.calcHist(images = image, channels = [0,1,2], mask = None, histSize = [8,8,8], ranges = [0, 256, 0, 256, 0, 256])
    hists_expanded = cv2.calcHist(images = image_expanded, channels = [0,1,2], mask = None, histSize = [8,8,8], ranges = [0, 256, 0, 256, 0, 256])

    # Compute the distance between the histograms using the chi-squared method.
    distance_histograms = cv2.compareHist(hists_cropped, hists_expanded, cv2.HISTCMP_CHISQR)

    ## 4) Edge Density
    edges = canny_detector(image = image_shrunken, sigma = 0.33)
    edge_density = np.sum(edges)/(edges.shape[0]*edges.shape[1])

    ## 5) Superpixel Straddeling
    #superpixel = superpixel_straddeling(image = image, width = width, height = height, x = x, y = y)

    ## 6) SD Depth
    sd_depth = np.std(depth_cropped)

    ## 7) SD Saliency
    sd_saliency = np.std(saliencyMap)

    ## 8) Distance between cropped and expanded depth map
    # Get the min and max values for histograms.

    MAX = max(np.max(depth_cropped), np.max(depth_expanded))
    MIN = min(np.min(depth_cropped), np.min(depth_expanded))

    hist_cropped_depth = np.histogram(np.array(depth_cropped).flatten(), bins=300, range = (MIN, MAX))[0]
    hist_expanded_depth = np.histogram(np.array(depth_expanded).flatten(), bins=300, range=(MIN, MAX))[0]

    distance_depth_histograms = euc_distance(X = hist_cropped_depth, Y = hist_expanded_depth)

    ## 9) Share of salient object larger than the twice the mean
    saliencyMap_image = saliency_map_estimation(image = image)
    mean_saliency_image = np.mean(saliencyMap_image)

    binary_saliency = saliencyMap_image > 2 * mean_saliency_image
    binary_saliency = np.sum(binary_saliency)/len(binary_saliency.flatten())



    return mean_depth, mean_saliency, distance_histograms, edge_density, sd_depth, sd_saliency, \
           distance_depth_histograms, binary_saliency



def get_features(label_dictionary, depth, images, beta):
    image_names = list(label_dictionary.keys())
    # Stupid hack that indicattes that the image loading does not work perfectly...
    all_features = []

    # Iterate over different images
    for index, name in enumerate(image_names):

        print(100*(index/len(image_names)))

        # Each image has positive and negative examples/windows that cover more than theta percent IoU
        for example in range(2):
            windows_example_list = len(label_dictionary[name][example])

            # Extract the features from each of these windows and save them as a list.
            for window in range(windows_example_list):
                mean_depth, mean_saliency, color_contrast, edge_density, sd_depth, sd_saliency, distance_depth_histograms, binary_saliency = extract_features(depth[index],
                                                                                            images[index],
                                                                                            label_dictionary[name][example][window],
                                                                                            beta = beta)

                features = (mean_depth, mean_saliency, color_contrast, edge_density, sd_depth, sd_saliency, distance_depth_histograms, binary_saliency)

                all_features.append(features)

    return all_features






