"""
not finished with hyperparameter tuning and architecture but
should work if its pasted in a google colab notebook
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cnn_model = Sequential()
# hyperparameters
embedding_size = 300
max_seq_len = 300
input_shape = (max_seq_len, embedding_size)
filters = [128, 128]
kernels = [3, 3]
hidden_dims = 250
learning_rate = 0.0005
batch_size = 32
epochs = 10
dataset_size = 20000

# Convolutional layers
cnn_model.add(Conv1D(filters[0], kernels[0], padding='valid', activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling1D(3))
cnn_model.add(Conv1D(filters[1], kernels[1], padding='valid', activation='relu'))
cnn_model.add(MaxPooling1D(3))

# Global max pooling layer
cnn_model.add(GlobalMaxPooling1D())

# Dense layers
cnn_model.add(Dense(hidden_dims, activation='relu'))
cnn_model.add(Dropout(0.75))
cnn_model.add(Dense(3, activation='sigmoid'))  # 3 output classes for negative, neutral, and positive sentiment

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
cnn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

cnn_model.summary()
print(tf.config.list_physical_devices('GPU'))

print('loading data')
X = np.load('samples/word_vectors-20k-300-300.npy')
y = np.load('samples/categories_words-20k-300-300.npy')
def words_to_categories(words):
    if words == 'positive':
        return [1, 0, 0]
    elif words == 'negative':
        return [0, 0, 1]
    elif words == 'neutral':
        return [0, 1, 0]

y = np.array([words_to_categories(i) for i in y])

print('loaded data')

print('splitting data')
print('dataset size: ', dataset_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('freeing memory')
del X
del y

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('fitting model')
cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=1/6, verbose=2, callbacks=[tensorboard_callback])

cnn_model.save('models/cnn_model.keras')

