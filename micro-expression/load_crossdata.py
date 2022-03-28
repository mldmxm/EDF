import h5py
import numpy as np
# VIS,CASME2
from collections import Counter
def load_crossdata():
    data_path_train = 'data/CrossCorpus_HOGTOP_P8_SMIC_NIR_112by112.mat'
    data_path_test = 'data/CrossCorpus_HOGTOP_P8_SMIC_VIS_112by112.mat'
    dict_data_train = h5py.File(data_path_train)
    dict_data_test = h5py.File(data_path_test)
    data_train = dict_data_train['SMIC_NIR_micro_feature']
    label_train = dict_data_train['SMIC_NIR_micro_label']
    data_test = dict_data_test['SMIC_VIS_micro_feature']
    label_test = dict_data_test['SMIC_VIS_micro_label']
    X_train= np.array(data_train).transpose()
    y_trian = np.array(label_train)
    X_test = np.array(data_test).transpose()
    y_test = np.array(label_test)
    temp1 = np.ones(np.shape(y_trian), dtype=np.int32)
    Y_train = (y_trian - temp1).reshape(np.shape(y_trian)[1],)
    temp2 = np.ones(np.shape(y_test), dtype=np.int32)
    Y_test = (y_test - temp2).reshape(np.shape(y_test)[1],)
    print(np.shape(X_train))
    print(np.shape(Y_train))
    print(np.unique(Y_train))

    for i in range(len(Y_test)):
        if Y_test[i]==0:
            Y_test[i]=1
        elif Y_test[i]==1:
            Y_test[i] = 0
    for i in range(len(Y_train)):
        if Y_train[i]==0:
            Y_train[i]=1
        elif Y_train[i]==1:
            Y_train[i] = 0
    print(Counter(Y_train))
    print(Counter(Y_test))
    # print(X_train)
    # print(Y_train)
    return X_train, X_test, Y_train, Y_test



