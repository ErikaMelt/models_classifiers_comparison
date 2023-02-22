from models.logistic_regression import train_model_lr
from models.random_forest import train_model_rf
from models.naive_bayes import train_model_nb
from models.data import load_data

# Load the data
X_train, y_train, X_test, y_test = load_data()

def select_model(model_name):
    if model_name == "Logistic Regression":
        train_model_lr(X_train, y_train, X_test, y_test)
    elif model_name == "Random Forest":
        train_model_rf()
    elif model_name == "Naive Bayes":
        train_model_nb()



