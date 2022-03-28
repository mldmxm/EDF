import numpy as np
import os
import cv2
def load_onepeople():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    people = '011'
    dataset = 'SAMM'

    for spiltdata in os.listdir(r"./data_leaveone_all/" + dataset + '/' + people + '/'):
        for pic_class in os.listdir(r"./data_leaveone_all/" + dataset + '/' + people + '/' + spiltdata + '/'):
            for pic in os.listdir(r"./data_leaveone_all/" + dataset + '/' + people + '/' + spiltdata + '/' + pic_class):

                img = cv2.imread(
                    './data_leaveone_all/' + dataset + '/' + people + '/' + spiltdata + '/' + pic_class + '/' + pic)
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


    return X_train, X_test, Y_train, Y_test