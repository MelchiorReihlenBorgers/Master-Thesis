import os
import numpy as np
import pandas as pd

def simulate_windows(N = 1):
    """
    :param N: Number of windwos to estimate. Default = 1
    :return: Three lists, with the top left coordinate, the width and the height of an estimated window
    """
    # TODO: The returns should depend on the input image. Thus, this should be a class, where the input
    # TODO: is the image from the DataLoad class.
    # TODO: Add Xdim, ydim as attributes in DataLoad
    width = np.random.randint(low = 500, high = 3000, size = 1)

    height = np.random.randint(low = 300, high = 2000, size = 1)

    x_coordinate = np.random.randint(high = 4032 - width, low = 0, size = 1)

    y_coordinate = np.random.randint(high = 3024 - height, low = 0, size = 1)

    return width, height, x_coordinate, y_coordinate



if os.path.exists(os.path.join(os.getcwd(), "test_annotation.csv")):
    annotation = pd.read_csv("test_annotation.csv",
                             header = None,
                             names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

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
    :return: List of boolean values indicating whether a window intersects with the annotated window.
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

    """
    # In case you want to print everything
    print("Width Simulated: {} \n Height Simulated: {} \n X Simulated: {} \n Y Simulated: {}".format(width, height, x, y))
    print("------------------------------------------------------------------------ \n------------------------------------------------------------------------")
    print(" Width Annotated: {} \n Height Annotated: {} \n X Annotated: {} \n Y Annotated: {}".format(annotation_width, annotation_height, annotation_x, annotation_y))
    print("------------------------------------------------------------------------ \n------------------------------------------------------------------------")
    print("Top Left Simulated: {} \n Bottom Right Simulated: {} \n Top Left Annotated: {} \n Bottom Right Annoted: {}".format(tl_sim, br_sim, tl_annotation, br_annotation))
    """
    intersection = True

    return intersection, width, height, x, y


a = does_intersect(annotation_x= 1, annotation_y=10, annotation_height=10, annotation_width = 10)

def compute_area():
    """
    Compute the area of a rectangle using the top left point.
    :return: Area of the square given
    """

    pass


def compute_overlap():
    """
    Get the top left coordinate of the overlap rectangle, the width and the height.
    """

    pass

def how_much_overlap():
    """
    Simulate the windwos, check whether they intersect (using simulate windows and does_intersect). If they do,
    check if they overlap by how much using the compute_area function to calculate the area of the overlapping part.
    :return: List of doubles between 0 and 1 indicating how much the simulated window and the annotation overlap.
    """

    pass


def compute_features():
    """
    Computes features over a window that covers the object (while clearing the threshold theta) to be used for
    the naive bayes algorithm.
    :return: One list per feature.
    """

    pass


if __name__ == "__main__":
    widths, heights, x, y = simulate_windows(N=10)
