import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plotting_windows(path, x, y, width, height, annotation_x, annotation_y, annotation_width, annotation_height):
    """
    Simple wrapper around pyplot patches and imshow
    """

    if os.path.exists(path):
        image = plt.imread(path)

    else:
        raise TypeError("Please provide an existing image path.")

    plt.figure()
    plt.imshow(image)

    ax = plt.gca()

    rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')

    ground_truth = Rectangle((annotation_x, annotation_y), annotation_width, annotation_height, linewidth=1, edgecolor='g', facecolor='none')

    ax.add_patch(ground_truth)
    ax.add_patch(rect)