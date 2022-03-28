import numpy as np
import os
import cv2
import matplotlib.image as mping
#   load data
def load_oederone():
    x_train, y_train, x_test, y_test = [], [], [], []
    people, dataset = 's03', 'SMIC'
    train_path_0 = "./data_leaveone_all/" + dataset + '/' + people + '/' + "u_train/0"
    train_path_1 = "./data_leaveone_all/" + dataset + '/' + people + '/' + "u_train/1"
    train_path_2 = "./data_leaveone_all/" + dataset + '/' + people + '/' + "u_train/2"
    train_0 = os.listdir(train_path_0)
    train_1 = os.listdir(train_path_1)
    train_2 = os.listdir(train_path_2)
    train_0.sort(key=lambda x: str(x[:-4]))
    train_1.sort(key=lambda x: str(x[:-4]))
    train_2.sort(key=lambda x: str(x[:-4]))

    # #
    test_path_0 = "./data_leaveone_all/" + dataset + '/' + people + '/' + "u_test/0"
    test_0 = os.listdir(test_path_0)
    test_0.sort(key=lambda x: str(x[:-4]))
    # print(test_0)
    for _, i in enumerate(test_0):
        if _ > 12:
            img = cv2.imread(test_path_0 + '/' + i)
            x_test.append(img)
            # print(x_test)
            y_test.append(np.int(0))

    # # #
    # test_path_1 = "./data_leaveone_all/" + dataset + '/' + people + '/' + "u_test/1"
    # test_1 = os.listdir(test_path_1)
    # test_1.sort(key=lambda x: str(x[:-4]))
    # # print(test_1)
    # for _, i in enumerate(test_1):
    #
    #     img = cv2.imread(test_path_1 + '/' + i)
    #     x_test.append(img)
    #     y_test.append(np.int(1))
    # # #
    #
    # test_path_2 = "./data_leaveone_all/" + dataset + '/' + people + '/' + "u_test/2"
    # test_2 = os.listdir(test_path_2)
    # test_2.sort(key=lambda x: str(x[:-4]))
    # # print(test_2)
    # for _, i in enumerate(test_2):
    #
    #     img = cv2.imread(test_path_2 + '/' + i)
    #     x_test.append(img)
    #     y_test.append(np.int(2))
    #

    for _, i in enumerate(train_0):
        img = cv2.imread(train_path_0 + '/' + i)
        x_train.append(img)
        y_train.append(np.int(0))

    for _, i in enumerate(train_1):
        img = cv2.imread(train_path_1 + '/' + i)
        x_train.append(img)
        y_train.append(np.int(1))

    for _, i in enumerate(train_2):
        img = cv2.imread(train_path_2 + '/' + i)
        x_train.append(img)
        y_train.append(np.int(2))











    # print(y_train)
    # print(np.shape(x_train))
    # print(np.shape(y_train))
    # print(np.shape(x_test))
    # print(np.shape(y_test))
    X_train = np.array(x_train)
    X_test = np.array(x_test)
    # print(y_train)
    Y_train = np.array(y_train)
    Y_test = np.array(y_test)


    return X_train, X_test, Y_train, Y_test





