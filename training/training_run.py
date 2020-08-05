import os
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from training.create_training_sample import create_training_sample
from training.extract_features import get_features

from training.Classification import Classification

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
                                                                                        K = 100, theta = 0.35, width_low = mean_annotation_width, height_low = mean_annotation_height)
# Load all images
data = DataLoad(path="/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis")
images, depth, radian = data.load_data(N = 50)

print("{} Images, {} Depths and {} Radians measures were loaded".format(len(images), len(depth), len(radian)))

# Extract the features from the samples and save them to create the input data for the algorithm.
# mean_saliency, mean_depth = extract_features()

all_features = get_features(label_dictionary, depth, images, beta = 0.1)



# Create one data frame with all features
N = len(all_features)
mean_depth = [all_features[i][0] for i in range(N)]
mean_saliency = [all_features[i][1] for i in range(N)]
color_distance = [all_features[i][2] for i in range(N)]
edge_density = [all_features[i][3] for i in range(N)]

#superpixel_straddeling = [all_features[i][4] for i in range(N)]


data = pd.DataFrame()
data["Mean_Saliency"] = mean_saliency
data["Mean_Depth"] = mean_depth
data["Color_Distance"] = color_distance
data["Edge_Density"] = edge_density
# data["Superpixel_Straddling"] = superpixel_straddeling
data["label"] = labels


################### ----------------------- ###################
###################    !START FROM HERE!    ###################
################### ----------------------- ###################

data.to_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/data.csv")

data = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/data.csv",
                   header = 0)

# Implement classification.
X = data.loc[:,["Mean_Saliency", "Mean_Depth", "Color_Distance", "Edge_Density"]]
y = data.loc[:,"label"]

nb = GaussianNB()
kmeans = KNeighborsClassifier()

methods = [ nb, kmeans ]

results = [ ]

for method in methods:
    classification = Classification(method=method, X=X, y=y)

    result = classification.evaluation("test")

    results.append(result)


