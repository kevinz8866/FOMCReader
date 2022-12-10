import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def map_age(age, low, high):
    if low <= age <= high:
        return f'{low}-{high}'
    elif age < low:
        return f'{low}-'
    else:
        return f'{high}+'

def map_prediction(pred): 
    """ This function maps the 2-col pd DataFrame of activation and sentiment back into emotion.
    """
    if pred['sentiment'] > 0:
        if pred['activation'] > 0:
            return 'happy'
        else:
            return 'neutral'
    else:
        if pred['activation'] > 0:
            return 'angry'
        else:
            return 'sad'

def read_df(name, label, CNN_gender, classes, low=None, high=None, continuous=False):
    y = pd.read_csv(f"input/cleaned_datasets/{name}/{label}_{name}.csv")
    if label == 'age' and continuous == False:
        y[label] = y[label].transform(lambda x:map_age(age=x, low=low, high=high))
    include_ind = y[label].isin(classes)
    if continuous:
        # Take out all ages below the low cutoff
        include_ind = y[label] >= low
    X = np.matrix(pd.read_csv(f"input/cleaned_datasets/{name}/X_{name}.csv")[include_ind])
    y = y[include_ind]
    if label != 'gender' and CNN_gender != 'both':
        gender = pd.read_csv(f"input/cleaned_datasets/{name}/gender_{name}.csv")[include_ind]
        gender_y = pd.concat([y, gender], axis=1)
        y[label] = gender_y[[label, 'gender']].apply(lambda x:x['gender'] + '_' + str(x[label]), axis=1)  
        ind = y[label].str.startswith(CNN_gender)
        X, y = X[ind], y[ind]
        y[label] = y[label].apply(lambda x:x.split('_')[-1])
    y['Cross Label'] = y[label].apply(lambda x:x + '_' + name)
    if label == 'age' and continuous:
        y[label] = y[label].astype(int)
    return X, y

def to_two_dimension(y, label):
    """ This function maps emotions to activation and sentiment. The mapping
    only goes from angry, sad, happy, and neutral.
    """
    activation_map = {'happy':1, 'angry':1, 'neutral':-1, 'sad': -1}
    sentiment_map = {'happy':1, 'angry':-1, 'neutral':1, 'sad': -1}
    y['sentiment'] = y[label].apply(lambda x:sentiment_map[x])
    y['activation'] = y[label].apply(lambda x:(activation_map[x]))
    return y[['sentiment', 'activation']]

def prepare_input(datasets, label, two_dimensional, classes, CNN_gender, age_low=None, age_high=None, continuous=False):
    """ This function prepares the datasets for training.
    """
    if two_dimensional and set(classes) != set(['angry', 'sad', 'neutral', 'happy']):
        raise ValueError("Two dimensional training can only be done with happy, angry, sad, and neutral emotions")
    # Training, testing prep
    X, y, y_cross, labs = [], [], [], []
    for dataset in datasets:
        X_dataset, y_df_dataset = read_df(dataset, label, CNN_gender, classes, age_low, age_high, continuous)
        if two_dimensional:
            y_df_dataset[['sentiment', 'activation']] = to_two_dimension(y_df_dataset, label)
            y_dataset = np.matrix(y_df_dataset[['sentiment', 'activation']])
        else:
            y_dataset = np.matrix(y_df_dataset[label])
        y_cross_dataset = np.matrix(y_df_dataset['Cross Label'])
        X.append(X_dataset), y.append(y_dataset), y_cross.append(y_cross_dataset), labs.append(np.matrix(y_df_dataset[label]))
    X = np.concatenate(X, axis=0)
    labs = np.concatenate(labs, axis=1).T
    if two_dimensional:
        y = np.concatenate(y, axis=0)
        lb = None
    elif continuous:
        y = np.ravel(y)
        lb = None
    else:
        y = np.concatenate(y, axis=1).T
        lb = LabelEncoder()
        y = to_categorical(lb.fit_transform(y))  
    y_cross = np.concatenate(y_cross, axis=1).T
    return X, y, y_cross, lb, labs