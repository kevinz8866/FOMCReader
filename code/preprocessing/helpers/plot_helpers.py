import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # to measure how good we are
import numpy as np
import pandas as pd
from helpers.data_helpers import *

def plot_loss(cnnhistory, name, path):
    """ This function plots the loss for the neural network training process.
    Args:
        cnnhistory ()
        name (str): the name 
    Returns:
        A plt plot saved in path, documenting the evolution of losses.
    
    """
    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{path}eval_figs/{name}.png')
    plt.clf()

def plot_cm(model, X_test, y_test, lb, name, path, classes, two_dimensional, separate=False):
    X_test = np.expand_dims(X_test, axis=2)
    if separate:
        # Obtain separate predictions, and map them back into original 
        # tone categories. 
        preds_sent = model['sentiment'].predict(X_test, batch_size=32, verbose=1).argmax(axis=1)
        preds_act = model['activation'].predict(X_test, batch_size=32, verbose=1).argmax(axis=1)
        preds_sent = (lb['sentiment'].inverse_transform((preds_sent)))
        preds_act = (lb['activation'].inverse_transform((preds_act)))
        preds = np.concatenate((np.matrix(preds_sent), np.matrix(preds_act)), axis=0).T
        predictions = process_2d_predictions(preds)
    else:
        preds = model.predict(X_test, batch_size=32, verbose=1)
        if two_dimensional:
            predictions = process_2d_predictions(preds)
        else:
            preds1 = preds.argmax(axis=1)
            predictions = (lb.inverse_transform((preds1)))
    cm = confusion_matrix(y_test, predictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    cm_plot = disp.plot()
    cm_plot.figure_.savefig(f'{path}eval_figs/{name}_cm.png')
    plt.clf()

def process_2d_predictions(predictions):
    predictions = pd.DataFrame(predictions, columns=['sentiment', 'activation'])
    predictions['preds'] = predictions.apply(map_prediction, axis=1) 
    return np.matrix(predictions['preds']).T