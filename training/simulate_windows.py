import numpy as np

def simulate_windows(N = 1, width_low=2500, height_low = 1900):
    """
    Helper function to simulate a window in an image.
    :param N: Number of windwos to estimate. Default = 1
    :param width_low, height_low: Lowest value for the width and height of the simulated windows. Default is set to 2500 and 1900, respectively. Can be set to mean of the annotations.
    :return: Three lists, with the top left coordinate, the width and the height of an estimated window
    """

    width = np.random.randint(low =width_low, high = 3000, size = 1)

    height = np.random.randint(low = height_low, high = 2000, size = 1)

    x_coordinate = np.random.randint(high = 4032 - width, low = 0, size = 1)

    y_coordinate = np.random.randint(high = 3024 - height, low = 0, size = 1)

    return width, height, x_coordinate, y_coordinate