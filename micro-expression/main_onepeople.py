"""
MNIST datasets demo for gcforest
Usage:
    define the model within scripts:
        python examples/demo_mnist.py
    get config from json file:
        python examples/demo_mnist.py --model examples/demo_mnist-gc.json
        python examples/demo_mnist.py --model examples/demo_mnist-ca.json
"""
import argparse
import numpy as np
import sys

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from load_3classdata import load_3data
from load_onepeople import load_onepeople
from order import load_oederone
from load_test import load_test
sys.path.insert(0, "../lib")
from collections import Counter
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from load_crossdata import load_crossdata
import matplotlib.image as mping
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 1
    ca_config["n_classes"] = 3
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 5, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 5, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 5, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 5, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 5, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 5, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config


def re_sample(X_train_temp, Y_train_temp):
    index = range(0, np.shape(Y_train_temp)[0])
    df = pd.DataFrame({'index': index, 'y_train': Y_train_temp})
    df_0 = df[df.iloc[:, -1] == 0]
    df_1 = df[df.iloc[:, -1] == 1]
    df_2 = df[df.iloc[:, -1] == 2]
    num_1_0 = len(df_0)-len(df_1)
    num_2_0 = len(df_0)-len(df_2)
    df_c1_sample = df_1.sample(n=num_1_0, replace=True, random_state=0)
    df_c2_sample = df_2.sample(n=num_2_0, replace=True, random_state=0)
    sample_1 = list(df_c1_sample['index'])
    sample_2 = list(df_c2_sample['index'])
    X_train_add_1 = X_train_temp[sample_1]
    Y_train_add_1 = Y_train_temp[sample_1]
    X_train_add_2 = X_train_temp[sample_2]
    Y_train_add_2 = Y_train_temp[sample_2]

    X_train_temp = np.vstack((X_train_temp, X_train_add_1))
    X_train_temp = np.vstack((X_train_temp, X_train_add_2))

    Y_train_temp = np.hstack((Y_train_temp, Y_train_add_1))
    Y_train_temp = np.hstack((Y_train_temp, Y_train_add_2))

    return X_train_temp, Y_train_temp

if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        config = get_toy_config()

    else:
        config = load_json(args.model)

    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.
    # X_train, X_test, y_train, y_test = load_onepeople()
    # X_train, X_test, y_train, y_test = load_3data()
    # X_train_temp, X_test, y_train_temp, y_test = load_oederone()

    X_train_temp, X_test, y_train_temp, y_test, source_picture = load_test()
    # X_train, X_test, y_train, y_test = load_crossdata()

    X_train_temp = X_train_temp[:, np.newaxis, :, :]

    X_test = X_test[:, np.newaxis, :, :]
    # X_train = X_train.reshape(np.shape(X_train)[0], -1)

    # print('Resampled dataset shape %s', Counter(y_train))
    # sm1 = SMOTE(ratio=1.0, random_state=None, n_neighbors=3, n_jobs=1)
    # X_res1, y_res1 = sm1.fit_sample(X_train, y_train)
    # print(np.shape(X_res1))
    # print('Resampled dataset shape %s', Counter(y_res1))
    # X_train_enc = gc.fit_transform(X_res1, y_res1)
    X_train, Y_train = re_sample(X_train_temp, y_train_temp)

    X_train_enc = gc.fit_transform(X_train, Y_train)
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('y-test', y_test)
    print('y-pred', y_pred)
    print('y-test', Counter(y_test))
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
    # gc.pyplot_picture(X_test, y_test, source_picture)

