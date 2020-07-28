def simulate_windows(N = 100):
    """
    :param N: Number of windwos to estimate. Default = 100
    :return: Three lists, with the bottom left coordinate, the width and the height of an estimated window
    """
    pass

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

def how_much_overlap():
    """
    Simulate the windwos, check whether they intersect (using simulate windows and does_intersect). If they do,
    check if they overlap by how much.
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