import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from training.create_training_sample import create_training_sample

if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_annotation.csv"):
    annotations = pd.read_csv("test_annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])


nrow = annotations.shape[0]

for row in range(nrow):
    annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[0,"X"], annotations.loc[0,"Y"], annotations.loc[0,"width"], annotations.loc[0,"height"]

    positive_examples, negative_examples = create_training_sample(annotation_x, annotation_y, annotation_width, annotation_height, K = 10000, theta = 0.4)



# Plotting
path = os.path.join(os.getcwd(), "test_image.jpg")
image = plt.imread(path)
plt.imshow(image)

ax = plt.gca()
rect = Rectangle((positive_examples[0][2], positive_examples[0][3]), positive_examples[0][0], positive_examples[0][1], linewidth=1, edgecolor='r', facecolor='none')
ground_truth = Rectangle((annotation_x, annotation_y), annotation_width, annotation_height, linewidth=1, edgecolor='g', facecolor='none')

ax.add_patch(ground_truth)
ax.add_patch(rect)



