import json

from models.data import load_data
from models.logistic_regression import train_model_lr
from models.naive_bayes import train_model_nb
from models.random_forest import train_model_rf


def get_model_metrics(n_clicks, model_selected):
    if n_clicks is None:
        return None
    else:
        if model_selected == "Logistic Regression":
            X_train, y_train, X_test, y_test = load_data()
            results = train_model_lr(X_train, y_train, X_test, y_test)
            return results
        elif model_selected == "Random Forest":
            # code to train random forest model and return results dictionary
            pass
        elif model_selected == "Naive Bayes":
            # code to train naive bayes model and return results dictionary
            pass
        return None

