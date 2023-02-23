from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from models.data import load_data
from models.plots import plot_heat_map, plot_ROC_Curve


def train_model_lr(data):
    """Train a logistic regression model on the training data."""
    # Load the data from the data store
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)

    plot_dict = plot_heat_map(cm)
    plot_ROC = plot_ROC_Curve(y_test, y_pred_prob)

    # Return the relevant parameters and evaluation metrics as a dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "cm": plot_dict,
        "roc": plot_ROC
    }

    return results
