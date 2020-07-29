import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def simulate_windows(N = 1):
    """
    Helper function to simulate a window in an image.
    :param N: Number of windwos to estimate. Default = 1
    :return: Three lists, with the top left coordinate, the width and the height of an estimated window
    """
    # TODO: The returns should depend on the input image. Thus, this should be a class, where the input
    # TODO: is the image from the DataLoad class.
    # TODO: Add Xdim, ydim as attributes in DataLoad
    # TODO: The window size should be based on the distribution of sizes of the annotations in the training data to make sure that the criterion is fullfilled multiple times
    width = np.random.randint(low = 1000, high = 3000, size = 1)

    height = np.random.randint(low = 600, high = 2000, size = 1)

    x_coordinate = np.random.randint(high = 4032 - width, low = 0, size = 1)

    y_coordinate = np.random.randint(high = 3024 - height, low = 0, size = 1)

    return width, height, x_coordinate, y_coordinate


def does_intersect(annotation_x, annotation_y, annotation_width, annotation_height):
    """
    Check whether two boxes overlap or not.
    The .csv exported from makesense.ai has the following columns:
        - Label
        - X of top left point
        - Y of top left point
        - width
        - height
        - Image name
        - Xdim image
        - Ydim image

    :param annotation: .csv file obtained from annotating the image on makesense.ai
    :return:    intersection: Bool that indicates whether the two rectangles intersect
                width, height, x, y: Width, height and xy coordinates of top left corner of the simulated rectangle
    """

    width, height, x, y = simulate_windows(N = 1)

    # Get the top left and bottom right corner of both the annotated window and the simulated window.
    tl_sim = (x, y)
    br_sim = (x + width, y + height)

    tl_annotation = (annotation_x, annotation_y)
    br_annotation = (annotation_x + annotation_width, annotation_y + annotation_height)

    # Code the conditions
    sim_right_of_annotation = tl_sim[0] >= br_annotation[0]
    sim_left_of_annotation = tl_annotation[0] >= br_sim[0]

    ## Be careful here. Images start at zero, zero in the top left corner -->  going to the bottom left corner means going up the y values.
    sim_above_annotation = br_sim[1] <= tl_annotation[1]
    sim_below_annotation = br_annotation[1] <= tl_sim[1]

    # If one rectangle is on left side of other
    if (sim_right_of_annotation or sim_left_of_annotation):
        intersection = False

    # If one rectangle is above other
    if (sim_above_annotation or sim_below_annotation):
        intersection = False


    intersection = True

    return intersection, width, height, x, y

# Do both rectangles intersect?
print(does_intersect(annotation_x= 1, annotation_y=10, annotation_height=10, annotation_width = 10)[0])

def compute_area(width, height):
    """
    Compute the area of a rectangle using the top left point.
    :return: Area of the rectangle given
    """
    area = width*height

    return area




def compute_overlap(x1, y1, width1, height1, x2, y2, width2, height2):
    """
    Get the top left coordinate of the overlap rectangle, the width and the height.

    :param x1, y1, width1, height1: xy coordinates, width and height of first rectangle
    :param x2, y2, width2, height2: xy coordinates, width and height of second rectangle

    :return: Size of the overlapping area
    """

    A1 = (x1, y1)
    A2 = (x1 + width1, y1 + height1)

    B1 = (x2, y2)
    B2 = (x2 + width2, y2 + height2)

    X = min(A2[0], B2[0]) - max(A1[0], B1[0])
    Y = min(A2[1], B2[1]) - max(A1[1], B1[1])

    overlapping_area = compute_area(X, Y)

    return overlapping_area

x1, y1, width1, height1 = 10, 10, 2, 20
x2, y2, width2, height2 = 11, 20, 2, 20


compute_overlap(x1, y1, width1, height1, x2, y2, width2, height2)



def compute_features():
    """
    Computes features over a window that covers the object (while clearing the threshold theta) to be used for
    the naive bayes algorithm.
    :return: One list per feature.
    """

    pass


if __name__ == "__main__":
    width, height, x, y = simulate_windows(N=1)

    import os

    if os.path.exists(os.path.join(os.getcwd(), "test_annotation.csv")):
        annotations = pd.read_csv("test_annotation.csv",
                                  header = None,
                                  names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

    annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[0,"X"], annotations.loc[0,"Y"], annotations.loc[0,"width"], annotations.loc[0,"height"]

    for _ in range(4):
        intersection, width, height, x, y  = does_intersect(annotation_x, annotation_y, annotation_width, annotation_height)

        # If they overlap, compute by how much they overlap
        if intersection:
            area_annotation = compute_area(width = annotation_width, height = annotation_height)
            area_window = compute_area(width = width, height = height)

            overlap = compute_overlap(x,y,width, height,
                                      annotation_x, annotation_y, annotation_width, annotation_height)

            # Is the overlap greater that 0.5 of the size of the sum of the annotation window and the simulated window?
            overlap_criterion = overlap/(area_annotation + area_window) > 0.5





        # Plotting
        path = os.path.join(os.getcwd(), "test_image.jpg")
        image = plt.imread(path)
        plt.imshow(image)

        ax = plt.gca()
        rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ground_truth = Rectangle((annotation_x, annotation_y), annotation_width, annotation_height, linewidth=1, edgecolor='g', facecolor='none')

        ax.add_patch(ground_truth)
        ax.add_patch(rect)



