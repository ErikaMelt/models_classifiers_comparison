from models.logistic_regression import train_model_lr
from models.naive_bayes import train_model_nb
from models.random_forest import train_model_rf


def get_model_metrics(n_clicks, model_selected, data):
    if n_clicks is None:
        return None
    else:
        if model_selected == "Logistic Regression":
            results = train_model_lr(data)
            return results
        elif model_selected == "Random Forest":
            results = train_model_rf(data)
            return results
        elif model_selected == "Naive Bayes":
            results = train_model_nb(data)
            return results
        return None

