import argparse
import numpy as np
import sys
import os
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


sys.path.insert(0, "../lib")
from collections import Counter
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


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
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    #
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config


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

    confuse_truey = np.array([])
    confuse_predy = np.array([])
    true_0, true_1, true_2, true_sample = 0, 0, 0, 0
    flase_0_1, flase_0_2, flase_1_0, flase_1_2, flase_2_0, flase_2_1 = 0, 0, 0, 0, 0, 0
    x_train, y_train, x_test, y_test = [], [], [], []

    for people in os.listdir(r"./data_leaveone_all/FULL/"):
        print(people)
        for spiltdata in os.listdir(r"./data_leaveone_all/FULL/" + people + '/'):
            for pic_class in os.listdir(r"./data_leaveone_all/FULL/" + people + '/' + spiltdata + '/'):
                for pic in os.listdir(r"./data_leaveone_all/FULL/" + people + '/' + spiltdata + '/' + pic_class):
                    img = cv2.imread(
                        './data_leaveone_all/FULL/' + people + '/' + spiltdata + '/' + pic_class + '/' + pic)
                    if spiltdata == 'u_train':
                        x_train.append(img)
                        y_train.append(np.int(pic_class))
                    else:
                        x_test.append(img)
                        y_test.append(np.int(pic_class))
        X_train = np.array(x_train)
        X_test = np.array(x_test)
        Y_train = np.array(y_train)
        Y_test = np.array(y_test)
        x_train, y_train, x_test, y_test = [], [], [], []

        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

        X_train_enc = gc.fit_transform(X_train, Y_train)

        y_pred = gc.predict(X_test, Y_test)

        acc = accuracy_score(Y_test, y_pred)
        print(people)
        print(Counter(Y_test))
        print('y_test', Y_test)
        print('y_pred', y_pred)
        print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

        for i in range(len(Y_test)):
            if Y_test[i] == 0:
                if y_pred[i] == 0:
                    true_0 += 1
                if y_pred[i] == 1:
                    flase_0_1 += 1
                if y_pred[i] == 2:
                    flase_0_2 += 1

            if Y_test[i] == 1:
                if y_pred[i] == 1:
                    true_1 += 1
                if y_pred[i] == 0:
                    flase_1_0 += 1
                if y_pred[i] == 2:
                    flase_1_2 += 1

            if Y_test[i] == 2:
                if y_pred[i] == 2:
                    true_2 += 1
                if y_pred[i] == 1:
                    flase_2_1 += 1
                if y_pred[i] == 0:
                    flase_2_0 += 1

    true_sample = true_0 + true_1 + true_2
    print('true_ne is', true_0)
    print('flase_0_1 is', flase_0_1)
    print('flase_0_2 is', flase_0_2)
    print('true_po is', true_1)
    print('flase_1_0 is', flase_1_0)
    print('flase_1_2 is', flase_1_2)
    print('true_sur is', true_2)
    print('flase_2_1 is', flase_2_1)
    print('flase_2_0 is', flase_2_0)
    print(true_sample)






