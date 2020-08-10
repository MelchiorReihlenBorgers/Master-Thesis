import pandas as pd


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

from training.Classification import Classification

# Read in the results from training_run.py
data = pd.read_csv("/media/melchior/Elements/MaastrichtUniversity/BISS/MasterThesis/data.csv",
                   header = 0)

# Implement classification.
X = data.loc[:,["Mean_Saliency", "Mean_Depth", "Color_Distance", "Edge_Density"]]
y = data.loc[:,"label"]

nb = GaussianNB()
knn = KNeighborsClassifier()
svc = SVC()

methods = [nb, knn, svc]

results = []
predictions = []


for index, method in enumerate(methods):
    classification = Classification(method=method, X=X, y=y)

    prediction_test = classification.predict("test")[0]
    prediction_train = classification.predict("train")[0]

    result_test = classification.evaluation("test")
    result_train = classification.evaluation("train")

    predictions.append([prediction_train, prediction_test])
    results.append([result_train, result_test])

# Make a pretty data frame for the prediction accuracy
method_names = ["Naive Bayes", "KNN", "SVM"]
results = pd.DataFrame(results, columns = ["In-Sample", "Out-Of-Sample"], index = method_names)


# Evaluation
classification = Classification("nb", X= X, y = y)
X_train, X_test, y_train, y_test = classification.create_split()

fit_svc = svc.fit(X_train, y_train)
fit_knn = knn.fit(X_train, y_train)
fit_nb = nb.fit(X_train, y_train)

plot_precision_recall_curve(fit_nb, X_test, y_test)
plt.title("Naive Bayes")
plot_precision_recall_curve(fit_svc,X_test, y_test)
plt.title("SVM")
plot_precision_recall_curve(fit_knn, X_test, y_test)
plt.title("KNN")
