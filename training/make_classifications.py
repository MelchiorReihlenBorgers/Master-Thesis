from sklearn.metrics import accuracy_score

def make_classifications(classifier, X_train, X_test, y_train, y_test):
    fit = classifier.fit(X_train, y_train)
    pred_in_sample = fit.predict(X_train)
    pred_out_of_sample = fit.predict(X_test)

    accuracy_out_of_sample = accuracy_score(y_true = y_test, y_pred = pred_out_of_sample)
    accuracy_in_sample = accuracy_score(y_true = y_train, y_pred = pred_in_sample)

    return classifier, accuracy_in_sample, accuracy_out_of_sample, pred_in_sample, pred_out_of_sample