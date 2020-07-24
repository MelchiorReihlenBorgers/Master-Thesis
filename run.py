"""
Main file of the project.
"""

import os
from DataLoad import DataLoad


# Load the data
path = os.getcwd()
data = DataLoad(path=os.getcwd())

n_total = len(data.extract_paths()[0])  # If you want to load all images.

images, depths, radians = data.load_data(N=3)


