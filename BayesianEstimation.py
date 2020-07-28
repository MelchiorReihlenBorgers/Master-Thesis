import numpy as np

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




def compute_overlap(A1, A2, B1, B2):
    """
    Get the top left coordinate of the overlap rectangle, the width and the height.

    :param A1: is the top left coordinate of the first rectangle
    :param A2: is the bottom right coordinate of the first rectangle
    :param B1: is the top left coordinate of the second rectangle
    :param B2: is the bottom right coordinate of the second rectangle
    """

    overlapping_area = max(0,(min(A2[1], B2[1])-max(A1[1],B1[1]))*(min(A2[0], B2[0])-max(A1[0], B1[0])))

    return overlapping_area

A1 = (1,3)
A2 = (3,1)
B1 = (2,4)
B2 = (4,2)

compute_overlap(A1, A2, B1, B2)



def compute_features():
    """
    Computes features over a window that covers the object (while clearing the threshold theta) to be used for
    the naive bayes algorithm.
    :return: One list per feature.
    """

    pass


if __name__ == "__main__":
    widths, heights, x, y = simulate_windows(N=10)
