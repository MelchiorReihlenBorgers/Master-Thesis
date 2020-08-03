import os
import pandas as pd
import numpy as np
import pandas as pd
import time

from training.plotting import  plotting_windows
from training.create_training_sample import create_training_sample
from training.extract_features import get_features

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


from DataLoad import DataLoad


# Load the .csv file with annotations (obtained from makesense.ai)
if os.path.exists("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv"):
    annotations = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/annotation.csv",
                              header = None,
                              names = ["Label", "X", "Y", "width", "height", "Image_Name", "Xdim", "Ydim"])

# Compute the mean annotation width and height and subtract the standard deviation once to
# use it as a lower bound in the sampling of random windows in the image.
mean_annotation_width = np.mean(annotations.loc[:,"width"]) - np.std(annotations.loc[:,"width"])
mean_annotation_height = np.mean(annotations.loc[:,"height"]) - np.std(annotations.loc[:,"height"])

# Save the number of rows in the annotations csv file.
# This is used in the subsequent loop to sample 100 random windows per annotation (one annotation per image so far...)
label_dictionary, labels, positive_examples, negative_examples = create_training_sample(annotations = annotations,
                                                                                        K = 100, theta = 0.4, width_low = mean_annotation_width, height_low = mean_annotation_height)

# Plotting
# Plot one example just to show how it works.
x, y, width, height = positive_examples[0][2], positive_examples[0][3], positive_examples[0][0], positive_examples[0][1]
annotation_x, annotation_y, annotation_width, annotation_height = annotations.loc[0,"X"], annotations.loc[0,"Y"], annotations.loc[0,"width"], annotations.loc[0,"height"]

plotting_windows("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/test_image.jpg",
                 x, y, width, height,
                 annotation_x, annotation_y, annotation_width, annotation_height)


data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")
images, depth, radian = data.load_data(N = 50)


# Extract the features from the samples and save them to create the input data for the algorithm.
# mean_saliency, mean_depth = extract_features()

all_features = get_features(label_dictionary, depth, images)



# Create one data frame with all features
N = len(all_features)
mean_depth = [all_features[i][0] for i in range(N)]
mean_saliency = [all_features[i][1] for i in range(N)]

data = pd.DataFrame()
data["Mean_Saliency"] = mean_saliency
data["Mean_Depth"] = mean_depth
data["label"] = labels

# Implement classification.
nb = GaussianNB()
X = data.loc[:,["Mean_Saliency", "Mean_Depth"]]
y = data.loc[:,"label"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

pred_in_sample = nb.fit(X_train, y_train)
pred_out_of_sample = nb.predict(X_test)

accuracy_out_of_sample = accuracy_score(y_true = y_test, y_pred = pred_out_of_sample)

print("The accuracy out of sample is: {:.2f}%".format(accuracy_out_of_sample*100))