import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import os, sys
import datetime
from helpers.data_helpers import *
from helpers.plot_helpers import *
from helpers.model_helpers import *
import argparse

if __name__ == "__main__":
    # Hyperparameters:
    include_gender = True # Whether to include gender in training labels
    lrs = [0.00001]
    epochs = 200
    separate = False # This script does not support separate = True
    two_dimensional = False # Indicating whether labels are two-dimensional (activation, sentiment)
    batch_size = 16
    k_thres = 5

    parser = argparse.ArgumentParser(description='Specification for neutral net.')
    parser.add_argument('-d', type=str, default=[], nargs='+',
                        help='a list of datasets')
    parser.add_argument('-l', type=str,
                        help='label')
    parser.add_argument('-c', type=str, nargs='+',
                        help='classes')
    parser.add_argument('-g', type=str,
                        help='gender')
    parser.add_argument('-dir', type=str, 
                        help='model directory')
    args = parser.parse_args()
    # Task-specific parameters
    # For gender:
    # classes = ['Male', 'Female']
    # For emotion:
    # classes = ['sad', 'angry', 'neutral', 'happy', 'disgust', 'fearful']
    datasets = args.d # Availdable datasets are RAVDESS, TESS, and CREMA-D
    label = args.l # Label can be 'gender', or 'emotion' (which are the original emotion labels)
    classes = args.c
    CNN_gender = args.g
    model_dir = args.dir
    continuous = False
    if classes == ['continuous']:
        continuous = True

    X, y, y_cross, lb, labs = prepare_input(datasets=datasets, label=label, two_dimensional=two_dimensional, 
                                            classes=classes, CNN_gender=CNN_gender, age_low=25, age_high=40, continuous=continuous)
    if two_dimensional:
        n_classes = 2
    else:
        n_classes = len(classes)
    if continuous: 
        kf = KFold(n_splits=5, random_state=10, shuffle=True)
    else: 
        kf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    # Set up file directory based on timestamp
    timestamp = datetime.datetime.today()
    path = 'output/models/' + model_dir + '/' + label + "_" + CNN_gender + '/'
    model_dir = os.path.join(os.getcwd(), path + 'saved_models/')
    fig_dir = os.path.join(os.getcwd(), path + 'eval_figs/')
    log_dir = os.path.join(os.getcwd(), path + 'logs/')
    os.makedirs(model_dir)
    os.makedirs(fig_dir)
    os.makedirs(log_dir)

    # Saving the label encoder
    if (two_dimensional == False or separate == True) and continuous == False:
        np.save(f'{path}/classes.npy', lb.classes_)
    # Initialize counter and dataframe documenting hyper params
    hyperparam_df = pd.DataFrame(columns=['Iteration Number', 'Include Gender', 
                                          'Learning Rate', "Dataset", 
                                          "Batch Size", "Epochs", "Two Dimensional", "Separate", 'CNN_gender',
                                          'Classes', 'Label'])
    print(y)

    for learning_rate in lrs:
        k = 0
        if continuous: 
            splits = kf.split(X)
        else:
            splits = kf.split(X, y_cross)
        for train_index, test_index in splits:
            if k < k_thres:
                if not continuous:
                    # Print label distribution
                    train_df = pd.DataFrame(y_cross[train_index], columns=['label'])
                    test_df = pd.DataFrame(y_cross[test_index], columns=['label'])
                    print(train_df['label'].value_counts())
                    print(test_df['label'].value_counts())
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = structure_model(learning_rate=learning_rate, two_dimensional=two_dimensional, 
                                        separate=False, n_classes=n_classes, continuous=continuous)
                table = pd.DataFrame(columns=["Name", "Type", "Shape"])
                for layer in model.layers:
                    table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
                with open(f"{path}/logs/analysis_{k}.log", "w") as log:
                    sys.stdout = log
                    cnnhistory = fit_model(X_train, X_test, y_train, y_test, model, epochs=epochs, batch_size=batch_size, verbose=1)
                sys.stdout = sys.__stdout__
                model_name = f'cnn_{k}.h5'
                model_json = model.to_json()
                # Save model and weights
                model_path = os.path.join(model_dir, model_name)
                model.save(model_path)
                with open(f'{model_dir}/model_{k}.json', "w") as json_file:
                    json_file.write(model_json)
                plot_loss(cnnhistory, model_name.split('.')[0], path)
                if not continuous:
                    plot_cm(model, X_test, labs[test_index], lb, model_name.split('.')[0], path, classes, two_dimensional)
                k += 1
                hyperparam_df = hyperparam_df.append({'Iteration Number':k, 
                                                      'Include Gender':include_gender, 
                                                      'Learning Rate':learning_rate, 
                                                      'Dataset':datasets, 'Batch Size':batch_size, 
                                                      'Epochs':epochs, 'Two Dimensional':two_dimensional, 
                                                      'Label':label,
                                                      'Separate':separate, 'CNN_gender':CNN_gender, 'Classes':classes}, ignore_index=True)
            
                hyperparam_df.to_csv(path + 'hyperparams.csv', index=False)



