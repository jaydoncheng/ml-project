import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

cnn_model = Sequential()
input_shape = (100, 100)
filters = [100, 100]
kernels = [5, 5]

# Convolutional layers
cnn_model.add(Conv1D(filters[0], kernels[0], activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling1D(3))
cnn_model.add(Conv1D(filters[1], kernels[1], activation='relu'))
cnn_model.add(MaxPooling1D(5))

# Global max pooling layer
cnn_model.add(GlobalMaxPooling1D())

# Dense layers
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(3, activation='softmax'))  # 3 output classes for negative, neutral, and positive sentiment

# Compile the model
optimizer = Adam(learning_rate=0.001)
cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

cnn_model.summary()


X_train = np.random.rand(100, 100, 100)
y_train = np.random.randint(low=-1, high=1, size=(100,3))
cnn_model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=1/6, verbose=2)
