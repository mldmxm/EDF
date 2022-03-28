import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
def load_3data():
    x = []
    y = []
    filename = '012'
    for filecla in filename:
        for pic in os.listdir(r"./data2/SMIC/" + filecla):
            img = cv2.imread('./data2/SMIC/' + filecla + '/' + pic)
            x.append(img)
            y.append(np.int(filecla))
    # print(np.shape(x))
    x = np.array(x)
    y = np.array(y)
    # print(np.shape(y))
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # X_test = X_test.reshape(X_test.shape[0], -1)
    print(np.unique(Y_test))
    return X_train, X_test, Y_train, Y_test


