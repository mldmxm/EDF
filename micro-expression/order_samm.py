import argparse
import numpy as np
import sys
import os
import cv2
import matplotlib.image as mping
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from load_3classdata import load_3data
import pandas as pd
sys.path.insert(0, "../lib")
from collections import Counter
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args

def re_sample(X_train_temp, Y_train_temp):
    index = range(0, np.shape(Y_train_temp)[0])
    df = pd.DataFrame({'index': index, 'y_train': Y_train_temp})
    df_0 = df[df.iloc[:, -1] == 0]
    df_1 = df[df.iloc[:, -1] == 1]
    df_2 = df[df.iloc[:, -1] == 2]
    num_1_0 = len(df_0)-len(df_1)
    num_2_0 = len(df_0)-len(df_2)
    df_c1_sample = df_1.sample(n=num_1_0, replace=True, random_state=41)
    df_c2_sample = df_2.sample(n=num_2_0, replace=True)
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
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
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
    y_true, ei_0, ei_1, ei_2, ei_3, ei_4, ei_5 = [], [], [], [], [], [], []
    Y_PRED=[]
    people = os.listdir(r"./data_leaveone_all/SAMM/")
    people.sort(key=lambda x: str(x))
    for _, one in enumerate(people):
        pic_class_train = os.listdir(r"./data_leaveone_all/SAMM/" + one + '/u_train')
        pic_class_train.sort(key=lambda x: str(x))
        pic_class_test = os.listdir(r"./data_leaveone_all/SAMM/" + one + '/u_test')
        pic_class_test.sort(key=lambda x: str(x))
        for i, pic_class in enumerate(pic_class_train):
            pic = os.listdir(r"./data_leaveone_all/SAMM/" + one + '/u_train/' + pic_class)
            pic.sort(key=lambda x: str(x[:-4]))
            for m, one_pic in enumerate(pic):
                img = cv2.imread(
                    './data_leaveone_all/SAMM/' + one + '/u_train/' + pic_class + '/' + one_pic)
                x_train.append(img)
                y_train.append(np.int(pic_class))

        for x, pictest_class in enumerate(pic_class_test):
            pic_test = os.listdir(r"./data_leaveone_all/SAMM/" + one + '/u_test/' + pictest_class)
            pic_test.sort(key=lambda x: str(x[:-4]))
            for n, one_pictest in enumerate(pic_test):
                img = cv2.imread(
                    './data_leaveone_all/SAMM/' + one + '/u_test/' + pictest_class + '/' + one_pictest)
                x_test.append(img)
                y_test.append(np.int(pictest_class))

        X_train_temp = np.array(x_train)
        X_test_temp = np.array(x_test)
        Y_train_temp = np.array(y_train)
        Y_test = np.array(y_test)
        x_train, y_train, x_test, y_test = [], [], [], []

        X_train_temp = X_train_temp[:, np.newaxis, :, :]
        X_test = X_test_temp[:, np.newaxis, :, :]

    #     #         ------------------------------------------
    #     #         get y_distribution
    #     for i in range(len(Y_test)):
    #         y_true.append(Y_test[i])
    #         if Y_test[i] == 0:
    #             ei_0.append(-1)
    #             ei_1.append(Y_test[i])
    #             ei_2.append(Y_test[i])
    #             ei_3.append(Y_test[i])
    #             ei_4.append(0)
    #             ei_5.append(0)
    #         elif Y_test[i] == 1:
    #             ei_0.append(0)
    #             ei_1.append(-1)
    #             ei_2.append(Y_test[i])
    #             ei_3.append(Y_test[i])
    #             ei_4.append(1)
    #             ei_5.append(0)
    #         elif Y_test[i] == 2:
    #             ei_0.append(1)
    #             ei_1.append(1)
    #             ei_2.append(-1)
    #             ei_3.append(1)
    #             ei_4.append(0)
    #             ei_5.append(1)
    # dataname = 'SAMM.csv'
    # result = pd.DataFrame(
    #     {'y_true': y_true, 'ei_0': ei_0, 'ei_1': ei_1, 'ei_2': ei_2, 'ei_3': ei_3, 'ei_4': ei_4, 'ei_5': ei_5})
    # result.to_csv('../result/' + dataname)

    # print(Y_train_temp)

        X_train, Y_train = re_sample(X_train_temp, Y_train_temp)
        X_train_enc = gc.fit_transform(X_train, Y_train)

        # X_train = X_train_temp.reshape(np.shape(X_train_temp)[0], -1)
        # print('Resampled dataset shape %s', Counter(Y_train_temp))
        # sm1 = SMOTE(ratio=1.0, random_state=None, n_neighbors=5, n_jobs=1)
        # X_res1, y_res1 = sm1.fit_sample(X_train, Y_train_temp)
        # print(np.shape(X_res1))
        # print('Resampled dataset shape %s', Counter(y_res1))
        # X_train_enc = gc.fit_transform(X_res1, y_res1)

        y_pred = gc.predict(X_test)
        Y_PRED.append(y_pred)
        y_true.append(Y_test)

        acc = accuracy_score(Y_test, y_pred)
        print(one)
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
    dataname = 'forest_SAMM.csv'
    result = pd.DataFrame(
        {'y_true': y_true, 'y-PRED': Y_PRED})
    result.to_csv('../result/' + dataname)






