"""
Model architecture for CNN:
    Embedding layer: 200x100 (200 words, 100 dimensions)
    Convolutional layer: 100 filters, kernel size 5
    Max pooling layer: pool size 2
    Dense layer: 128 units
    Output layer: 1 unit, sigmoid activation
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

cnn_model = Sequential()
# hyperparameters
embedding_size = 100
max_seq_len = 100
input_shape = (max_seq_len, embedding_size)
filters = [150, 150]
kernels = [5, 5]
hidden_dims = 250

# Convolutional layers
cnn_model.add(Conv1D(filters[0], kernels[0], padding='valid', activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling1D(5))
cnn_model.add(Conv1D(filters[1], kernels[1], padding='valid', activation='relu'))
cnn_model.add(MaxPooling1D(5))

# Global max pooling layer
cnn_model.add(GlobalMaxPooling1D())

# Dense layers
cnn_model.add(Dense(hidden_dims, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(3, activation='softmax'))  # 3 output classes for negative, neutral, and positive sentiment

# Compile the model
optimizer = Adam(learning_rate=0.001)
cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

cnn_model.summary()
print(tf.config.list_physical_devices('GPU'))

print('loading data')
X = np.load('word_vectors.npy')
y = np.load('categories.npy')
print('loaded data')

print('splitting data')
dataset_size = 10000
print('dataset size: ', dataset_size)
X_train, X_test, y_train, y_test = train_test_split(X[:dataset_size], y[:dataset_size], test_size=0.2, random_state=42)
print('split data')
print('freeing memory')
del X
del y

print('fitting model')
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=1/6, verbose=2)
