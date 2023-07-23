#NN.py
#Apply a neural network to a given dataset

#Imports
#Internal Libraries
import pdb
#External Libraries (pip)
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping
import keras.utils

def NN_algorithm(x, y):
    '''Use a neural network on the given data'''
    yb = keras.utils.to_categorical(y==1, 2)

    #Create Neural Network
    model = Sequential()
    model.add(Input(shape=(29,)))
    model.add(Dense(units=400, activation='relu', name="hidden1"))
    model.add(Dense(units=2, activation="softmax", name="output"))
    model.summary()

    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=[Precision(), Recall()])

    # Add optional callbacks
    callback = EarlyStopping(
        monitor='loss',
        min_delta=1e-4,
        patience=10,
        verbose=1)

    # Train the network
    history = model.fit(x[200000:, :], yb[200000:],
        epochs=1000,
        batch_size=3000,
        callbacks=[callback],
        verbose=1)

    # Test the network
    metrics = model.evaluate(x, yb, verbose=0)
    print(f'Precision = {metrics[1]:0.4f} | Recall = {metrics[2]:0.4f}')

    y = np.argmax(model.predict(x, verbose=0), axis=1)  # choose the output neuron with the highest value as the predicted output

