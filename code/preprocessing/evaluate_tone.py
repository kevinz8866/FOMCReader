from keras.models import model_from_json
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from helpers.data_helpers import *
import argparse
import glob
import os
import statistics
import random
import ast

def evaluate(model_dir, model_name, json_name, lb_path, X, continuous, outcome, i):
    """ Load the model and evaluate
    """
    json_file = open(model_dir + json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir + model_name)
    # evaluate loaded model on test data
    preds = loaded_model.predict(X, 
                                 batch_size=32, 
                                 verbose=1)
    if not continuous:
        lb = LabelEncoder()
        lb.classes_ = np.load(lb_path, allow_pickle=True)
        if outcome == 'emotion':
            cols = [emotion + f'_{i}' for emotion in lb.inverse_transform((range(preds.shape[1])))]
            return pd.DataFrame(preds, columns=cols)
        else:    
            preds1 = preds.argmax(axis=1)
            predictions = (lb.inverse_transform((preds1)))
            return pd.DataFrame(predictions, columns=[outcome + f'_{i}'])
    else:
        return pd.DataFrame(np.ravel(preds.T), columns=[outcome + f'_{i}'])

if __name__ == "__main__":
    df = pd.read_csv("output/audio_features/features.csv")
    dates = df['date']
    indices = df['index']
    X = np.matrix(df.drop(columns=['date', 'index']))
    X = np.expand_dims(X, axis=2)

    classification_results = []
    for name in ['emotion_Female', 'emotion_Male', 'gender_both']:
        model = 'model/CNN/' + name
        predictions = {}
        hyperparams = pd.read_csv(f'{model}/hyperparams.csv')
        outcome = hyperparams['Label'][0]
        classes = ast.literal_eval(hyperparams['Classes'][0])
        gender = hyperparams['CNN_gender'][0]
        model_dir = f'{model}/saved_models/'
        if 'continuous' in hyperparams['Classes'][0]:
            continuous = True
        else:
            continuous = False
        for i in range(len(hyperparams['Iteration Number'])):
            print(f"Evaluating conference data based on {outcome}_{gender}_{i} model")
            model_name = f'cnn_{i}.h5'
            json_name = f'model_{i}.json'
            lb_path = f'{model}/classes.npy'
            predictions[outcome + f'_{i}'] = evaluate(model_dir, model_name, json_name, lb_path, X, continuous, outcome, i)
            print(f"Evaluating conference data based on {outcome}_{gender}_{i} model")
        classification = pd.concat(list(predictions.values()), axis=1)
        if outcome == 'emotion':
            emotion_classes = classes
            labels = {emotion + name:[emotion + f"_{i}" for i in range(len(hyperparams['Iteration Number']))] for emotion in classes}
            for emotion in labels.keys():
                classification.loc[:, emotion] = classification[labels[emotion]].mean(axis=1)
            classification = classification[labels.keys()]
        else: 
            classification.loc[:, outcome] = classification[list(predictions.keys())].apply(lambda x:random.choice(statistics.multimode(x)), axis=1)
            classification = classification[[outcome]]
        classification_results.append(classification)

    classification_results = pd.concat(classification_results, axis=1)
    for emotion in emotion_classes:
        classification_results[emotion] = classification_results.apply(lambda x: x[f'{emotion}emotion_{x["gender"]}'], axis=1)
    classification_results['date'] = dates
    classification_results['index'] = indices

    classification_results[['date', 'index'] + emotion_classes].sort_values(by=['date', 'index']).to_csv('output/tones.csv', index=False)
