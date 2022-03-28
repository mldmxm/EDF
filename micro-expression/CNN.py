from load_crossdata import load_crossdata
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
def re_sample(X_train_temp, Y_train_temp):
    index = range(0, np.shape(Y_train_temp)[0])
    df = pd.DataFrame({'index': index, 'y_train': Y_train_temp})
    df_0 = df[df.iloc[:, -1] == 0]
    df_1 = df[df.iloc[:, -1] == 1]
    df_2 = df[df.iloc[:, -1] == 2]
    num_1_0 = len(df_0)-len(df_1)
    num_2_0 = len(df_0)-len(df_2)
    df_c1_sample = df_1.sample(n=num_1_0, replace=True)
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


if __name__ == '__main__':
    X_train_temp, X_test, y_train_temp, y_test = load_crossdata()
    X_train, Y_train = re_sample(X_train_temp, y_train_temp)
    clf = SVC()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print('y-test', Counter(y_test))
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
    print("Test f1 of GcForest = {:.2f} %".format(f1 * 100))

    print('--------------')
    clf = AdaBoostClassifier(n_estimators=500)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Test Accuracy of ada = {:.2f} %".format(acc * 100))
    print("Test f1 of ada = {:.2f} %".format(f1 * 100))

