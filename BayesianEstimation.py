import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


mport os

if os.path.exists(os.path.join(os.getcwd(), "test_annotation.csv")):
    annotations = pd.read_csv("test_annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[0,"X"], annotations.loc[0,"Y"], annotations.loc[0,"width"], annotations.loc[0,"height"]


# Create some space to safe the positive and negative examples (i.e. windows that overlap enough and windows that do not)
positive_examples = []
negative_examples = []

for _ in range(1000):
    intersection, width, height, x, y  = does_intersect(annotation_x, annotation_y, annotation_width, annotation_height)

    # If they overlap, compute by how much they overlap
    if intersection:
        area_annotation = compute_area(width = annotation_width, height = annotation_height)
        area_window = compute_area(width = width, height = height)

        overlap = compute_overlap(x,y,width, height,
                                  annotation_x, annotation_y, annotation_width, annotation_height)

        # Is the overlap greater that 0.5 of the size of the sum of the annotation window and the simulated window?
        overlap_criterion = overlap/(area_annotation + area_window) > 0.2

        if overlap_criterion:
            positive_examples.append(np.array([width, height, x, y]))

        else:
            negative_examples.append(np.array([width, height, x, y]))


# Plotting
path = os.path.join(os.getcwd(), "test_image.jpg")
image = plt.imread(path)
plt.imshow(image)

ax = plt.gca()
rect = Rectangle((positive_examples[0][2], positive_examples[0][3]), positive_examples[0][0], positive_examples[0][1], linewidth=1, edgecolor='r', facecolor='none')
ground_truth = Rectangle((annotation_x, annotation_y), annotation_width, annotation_height, linewidth=1, edgecolor='g', facecolor='none')

ax.add_patch(ground_truth)
ax.add_patch(rect)



