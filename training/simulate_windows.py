import numpy as np

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
    width = np.random.randint(low = 2500, high = 3000, size = 1)

    height = np.random.randint(low = 1900, high = 2000, size = 1)

    x_coordinate = np.random.randint(high = 4032 - width, low = 0, size = 1)

    y_coordinate = np.random.randint(high = 3024 - height, low = 0, size = 1)

    return width, height, x_coordinate, y_coordinate