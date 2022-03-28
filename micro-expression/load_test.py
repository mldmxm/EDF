import numpy as np
import os
import cv2
import matplotlib.image as mping
#   load data
def load_test():
    x_train, y_train, x_test, y_test = [], [], [], []
    dataset = 'test2'
    train_path_0 = "./test/" + dataset + '/' + "u_train/0"
    train_path_1 = "./test/" + dataset + '/' + "u_train/1"
    train_path_2 = "./test/" + dataset + '/' + "u_train/2"

    train_0 = os.listdir(train_path_0)
    train_1 = os.listdir(train_path_1)
    train_2 = os.listdir(train_path_2)
    train_0.sort(key=lambda x: str(x[:-4]))
    train_1.sort(key=lambda x: str(x[:-4]))
    train_2.sort(key=lambda x: str(x[:-4]))

    # # # # #
    # test_path_0 = "./test/" + dataset + '/' + "u_test/0"
    # test_0 = os.listdir(test_path_0)
    # test_0.sort(key=lambda x: str(x[:-4]))
    # # print(test_0)
    # for _, i in enumerate(test_0):
    #     img = cv2.imread(test_path_0 + '/' + i)
    #     x_test.append(img)
    #     # print(x_test)
    #     y_test.append(np.int(0))

    #
    # test_path_1 = "./test/" + dataset + '/' + "u_test/1"
    # test_1 = os.listdir(test_path_1)
    # test_1.sort(key=lambda x: str(x[:-4]))
    # # print(test_1)
    # for _, i in enumerate(test_1):
    #     # img = cv2.imread(test_path_1 + '/' + i)
    #     img = cv2.imread(test_path_1 + '/' + i)
    #     x_test.append(img)
    #     y_test.append(np.int(1))
    # # # # # # #
    # #
    test_path_2 = "./test/" + dataset + '/' + "u_test/2"
    test_2 = os.listdir(test_path_2)
    test_2.sort(key=lambda x: str(x[:-4]))
    # print(test_2)
    for _, i in enumerate(test_2):
        img = cv2.imread(test_path_2 + '/' + i)
        x_test.append(img)
        y_test.append(np.int(2))


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

    #  get source picture
    source_picture =[]
    source_path = "./test/" + dataset + '/' + "u_test/source"
    source = os.listdir(source_path)
    source.sort(key=lambda x: str(x[:-4]))
    # # print(test_2)
    for _, i in enumerate(source):
        img = cv2.imread(source_path + '/' + i)
        source_picture.append(img)








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
    source_picture = np.array(source_picture)



    return X_train, X_test, Y_train, Y_test, source_picture





