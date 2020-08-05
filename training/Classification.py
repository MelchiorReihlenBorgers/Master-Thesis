"""
Class to implement different classificatoin schemes and to make predictions.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Classification(object):

    def __init__(self, method, X, y):
        """
        :param method: Object, initializing the classifier.
        :param X: Predictors/Features
        :param y: Dependent variable
        """
        self.X = X
        self.y = y
        self.method = str(method)


    def create_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state = 0)

        return X_train, X_test, y_train, y_test

    def predict(self, kind):
        """
        Predict given the method and kind (train or test)
        :param kind: String indicating whether to use train or test data for predictions.
        :return: Predicted values and probabilities.
        """
        X_train, X_test, y_train, y_test = self.create_split()

        fit = eval(self.method + ".fit(X_train, y_train)")


        if kind == "train":
            predict = fit.predict(X_train)
            y_true = y_train

        else:
            predict = fit.predict(X_test)
            y_true = y_test

        return predict, y_true

    def evaluation(self, kind):
        """
        Evaluate classifier.
        :param kind:
        :return:
        """

        if kind == "train":
            predict, y_true = self.predict(kind = kind)

        else:
            predict, y_true = self.predict(kind=kind)

        accuracy = accuracy_score(y_true=y_true, y_pred=predict)

        return accuracy
