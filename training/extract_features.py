import os
import matplotlib.pyplot as plt

from training.saliency_map_estimation import saliency_map_estimation

def extract_features(image_name, examples_dict):
    """
    Extract features from the different windows selected.

    1. Compute mean saliency
    2. Compute mean depth

    The intention is that this is to be used in an iterative process where the function is called upon iteratively.

    :param image: Image to be used
    :param examples_dict: Dictionary with the positive and negative examples created in training_run.py
    """
    path_image = os.path.join(os.getcwd(), image_name)
    image = plt.imread(path_image)

    saliency_map_estimation(image = image)

    # TODO: Connect this to the DataLoad() module. You should not have two different data loads. Maybe sth along the lines of find the path that ends on image_name and use that
    # TODO: picture and depth.



