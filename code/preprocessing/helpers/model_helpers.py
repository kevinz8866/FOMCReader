from tensorflow.keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError
import numpy as np

def structure_model(learning_rate, n_classes, two_dimensional, separate, continuous):
    '''
    Structure a model that has two layers of conv layers, followed by a pooling layer,
    and two more conv layers
    '''
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(180, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same',))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    opt = optimizers.RMSprop(learning_rate=learning_rate, decay=1e-6)
    if two_dimensional:
        model.add(Dense(n_classes))
        if separate:
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        else: 
            model.add(Activation('linear'))
            model.compile(loss=MeanSquaredError(), optimizer=opt)
    elif continuous:
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss=MeanSquaredError(), optimizer=opt)
    else:
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def fit_model(X_train, X_test, y_train, y_test, model, epochs=700, batch_size=16, verbose=0):
    '''Fit the model by first reshaping training and testing data.'''
    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)
    return model.fit(x_traincnn, y_train, batch_size=batch_size, 
                     epochs=epochs, verbose=verbose, 
                     validation_data=(x_testcnn, y_test))