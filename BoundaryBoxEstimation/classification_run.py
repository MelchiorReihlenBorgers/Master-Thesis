import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import scale
from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt

## Just a bool to check whether to save results (Y) as csv or not (anything else)
save_bool = input(str("Do you want to save the files?"))


####### I: Preprocessing #######

# Read in the results from training_run.py
data = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/data.csv",
                   header = 0)

# Save column naames
col_names = ["Mean_Depth", "Mean_Saliency", "Color_Contrast", "Edge_Density", "SD_Depth", "SD_Saliency",
             "Depth_Contrast", "Binary_Saliency"]

X = data.loc[:,col_names]

# Downsize the color and depth distance
X.loc[:,["Color_Contrast", "Depth_Contrast"]] = X.loc[:,["Color_Contrast", "Depth_Contrast"]]/1000

# Scale the data by removing the mean and the standard deviation.
X = scale(X)

y = data.loc[:,"label"]

# Create train-test split

## Data with all columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

## Data without information about depth
X_train_ND, X_test_ND = np.delete(arr = X_train, obj = [0,4,6], axis = 1), np.delete(arr = X_test, obj = [0,4,6], axis = 1)

####### II: Classification #######

# Define Classifiers
nb = GaussianNB()
knn = KNeighborsClassifier()
svc = SVC(probability= True)

## Fit Classifiers without depth:
fit_nb_ND = nb.fit(X_train_ND, y_train)
fit_knn_ND = knn.fit(X_train_ND, y_train)
fit_svc_ND = svc.fit(X_train_ND, y_train)

# Predict with Classifiers
## Save methods in dict to iterate over them.
methods = {"Naive Bayes":nb,
           "KNN": knn,
           "SVM": svc}

## With Depth:
accuracies = []
precisions = []


for method_name, method in methods.items():

    # Fit model
    fit = method.fit(X_train, y_train)

    # Predict in-sample
    prediction_in_sample = fit.predict(X_train)
    accuracy_in_sample = accuracy_score(y_true = y_train, y_pred = prediction_in_sample)
    maP_in_sample = average_precision_score(y_true = y_train, y_score = prediction_in_sample)

    # Predict out-of-sample
    prediction_out_of_sample = fit.predict(X_test)
    accuracy_out_of_sample = accuracy_score(y_true=y_test, y_pred=prediction_out_of_sample)
    maP_out_of_sample = average_precision_score(y_true=y_test, y_score=prediction_out_of_sample)

    accuracy = (accuracy_in_sample, accuracy_out_of_sample)
    precision = (maP_in_sample, maP_out_of_sample)

    accuracies.append(accuracy)
    precisions.append(precision)

accuracies = pd.DataFrame(accuracies, columns = ["Train", "Test"], index = [method for method, x in methods.items()])
precisions = pd.DataFrame(precisions, columns = ["Train", "Test"], index = [method for method, x in methods.items()])

"""
accuracies.plot(kind = ""bar")
plt.xticks(rotation = 0)
plt.title("Train vs. Test Accuracy")
plt.show()

precisions.plot(kind = "bar")
plt.xticks(rotation = 0)
plt.title("Train vs. Test mAP")
plt.show()

if save_bool == "Y":
    accuracies.to_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/Results Master Thesis/Boundary Box Estimation/With Depth/Accuracies_Depth.csv")
    precisions.to_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/Results Master Thesis/Boundary Box Estimation/With Depth/Precisions_Depth.csv")
"""
## Without Depth:
accuracies_ND = []
precisions_ND = []
poos = []

for method_name, method in methods.items():
    # Fit model
    fit = method.fit(X_train_ND, y_train)

    # Predict in-sample
    prediction_in_sample_ND = fit.predict(X_train_ND)
    accuracy_in_sample_ND = accuracy_score(y_true = y_train, y_pred = prediction_in_sample_ND)
    maP_in_sample_ND = average_precision_score(y_true = y_train, y_score = prediction_in_sample_ND)

    # Predict out-of-sample
    prediction_out_of_sample_ND = fit.predict(X_test_ND)
    accuracy_out_of_sample_ND = accuracy_score(y_true=y_test, y_pred=prediction_out_of_sample_ND)
    maP_out_of_sample_ND = average_precision_score(y_true=y_test, y_score=prediction_out_of_sample_ND)

    accuracy = (accuracy_in_sample_ND, accuracy_out_of_sample_ND)
    precision = (maP_in_sample_ND, maP_out_of_sample_ND)


    poos.append(prediction_out_of_sample_ND)
    accuracies_ND.append(accuracy)
    precisions_ND.append(precision)

accuracies_ND = pd.DataFrame(accuracies_ND, columns = ["Train", "Test"], index = [method for method, x in methods.items()])
precisions_ND = pd.DataFrame(precisions_ND, columns = ["Train", "Test"], index = [method for method, x in methods.items()])

accuracies_ND.plot(kind = "bar")
plt.xticks(rotation = 0)
plt.title("Train vs. Test Accuracy")
plt.savefig("/media/melchior/Elements/MaastrichtUniversity/BISS/Results Master Thesis/Boundary Box Estimation/Without Depth/Accuracy_NoDepth.jpg")
#plt.show()

precisions_ND.plot(kind = "bar")
plt.xticks(rotation = 0)
plt.title("Train vs. Test mAP")
plt.show()


if save_bool == "Y":
    accuracies_ND.to_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/Results Master Thesis/Boundary Box Estimation/Without Depth/Accuracies_No_Depth.csv")
    precisions_ND.to_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/Results Master Thesis/Boundary Box Estimation/Without Depth/Precisions_No_Depth.csv")

"""
####### III: Precision Recall Plots #######
# Get precision of random classifier
random_classifier = np.sum(y_test)/len(y_test)

## With Depth:
for method_name, method in methods.items():
    fit = method.fit(X_train, y_train)
    plot_precision_recall_curve(method, X_test, y_test)
    plt.title("{} - PR Plot (incl. depth)".format(method_name))
    plt.axhline(random_classifier, color="red", linestyle="--")
    plt.show()

## Without Depth:
for method_name, method in methods.items():
    fit = method.fit(X_train_ND, y_train)
    plot_precision_recall_curve(method, X_test_ND, y_test)
    plt.title("{} - PR Plot (excl. depth)".format(method_name))
    plt.axhline(random_classifier, color="red", linestyle="--")
    plt.show()

"""