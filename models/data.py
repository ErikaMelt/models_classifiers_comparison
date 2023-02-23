from keras.datasets import imdb
import numpy as np


from sklearn.model_selection import train_test_split

def load_data(num_words=2000, test_size=0.2):
    """Load the IMDb movie review dataset and preprocess it."""
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=test_size, random_state=42)

    # Convert the reviews to a matrix
    X_train = vectorize_sequences(X_train)
    X_test = vectorize_sequences(X_test)

    # Convert the labels to binary format
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    return X_train, y_train, X_test, y_test



def vectorize_sequences(sequences, dimension=2000):
    """Convert the reviews to a matrix."""
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
