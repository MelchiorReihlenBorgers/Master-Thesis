import numpy as np

def simulate_windows(N = 100):
    """
    :param N: Number of windwos to estimate. Default = 100
    :return: Three lists, with the top left coordinate, the width and the height of an estimated window
    """
    # TODO: The returns should depend on the input image. Thus, this should be a class, where the input
    # TODO: is the image from the DataLoad class.
    # TODO: Add Xdim, ydim as attributes in DataLoad
    widths = [np.random.randint(low = 500, high = 3000, size = 1) for _ in range(N)]

    heights = [np.random.randint(low = 300, high = 2000, size = 1) for _ in range(N)]

    x_coordinate = [np.random.randint(high = 4032 - widths[i], low = 0, size = 1) for i in range(N)]

    y_coordinate = [np.random.randint(high = 3024 - heights[i], low = 0, size = 1) for i in range(N)]

    return widths, heights, x_coordinate, y_coordinate


widths, heights, x, y = simulate_windows(N = 10)

def does_intersect(annotation):
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

    pass

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