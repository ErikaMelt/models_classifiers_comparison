from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_model_lr(X_train, y_train, X_test, y_test):
    """Train a logistic regression model on the training data."""
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Return the trained model and evaluation metrics
    return model, accuracy, precision, recall

    return lr
