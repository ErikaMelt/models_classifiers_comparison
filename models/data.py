from keras.datasets import imdb
import  numpy as np

def load_data(num_words=2000):
    """Load the IMDb movie review dataset and preprocess it."""
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

    # Convert the reviews to a matrix
    X_train = vectorize_sequences(train_data)
    X_test = vectorize_sequences(test_data)

    # Convert the labels to binary format
    y_train = train_labels.astype('float32')
    y_test = test_labels.astype('float32')

    return X_train, y_train, X_test, y_test

def vectorize_sequences(sequences, dimension=2000):
    """Convert the reviews to a matrix."""
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
