# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets.
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng.
"""
import numpy as np
import os
import cv2
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
LOGGER = get_logger('gcforest.cascade.cascade_classifier')


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


class CascadeClassifier(object):
    def __init__(self, ca_config):
        """
        Parameters (ca_config)
        ----------
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in early_stopping_rounds, the cascade level will stop automatically growing
        max_layers: int
            maximum number of cascade layers allowed for exepriments, 0 means use Early Stoping to automatically find the layer number
        n_classes: int
            Number of classes
        est_configs:
            List of CVEstimator's config
        look_indexs_cycle (list 2d): default=None
            specification for layer i, look for the array in look_indexs_cycle[i % len(look_indexs_cycle)]
            defalut = None <=> [range(n_groups)]
            .e.g.
                look_indexs_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3; layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermidiate results in data_save_dir
            if data_save_rounds = 0, then no savings for intermidiate results
        """
        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        # self.data_save_dir = self.get_value("data_save_dir", None, basestring)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.opt_layer_num = -1
        # LOGGER.info("\n" + json.dumps(ca_config, sort_keys=True, indent=4, separators=(',', ':')))

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="cascade")

    def _set_estimator(self, li, ei, est):
        if li not in self.estimator2d:
            self.estimator2d[li] = {}
        self.estimator2d[li][ei] = est

    def _get_estimator(self, li, ei):
        return self.estimator2d.get(li, {}).get(ei, None)

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def _check_look_indexs_cycle(self, X_groups, is_fit):
        # check look_indexs_cycle
        n_groups = len(X_groups)
        if is_fit and self.look_indexs_cycle is None:
            look_indexs_cycle = [list(range(n_groups))]
        else:
            look_indexs_cycle = self.look_indexs_cycle
            for look_indexs in look_indexs_cycle:
                if np.max(look_indexs) >= n_groups or np.min(look_indexs) < 0 or len(look_indexs) == 0:
                    raise ValueError("look_indexs doesn't match n_groups!!! look_indexs={}, n_groups={}".format(
                        look_indexs, n_groups))
        if is_fit:
            self.look_indexs_cycle = look_indexs_cycle
        return look_indexs_cycle

    def _check_group_dims(self, X_groups, is_fit):
        if is_fit:
            group_starts, group_ends, group_dims = [], [], []
        else:
            group_starts, group_ends, group_dims = self.group_starts, self.group_ends, self.group_dims
        n_datas = X_groups[0].shape[0]
        X = np.zeros((n_datas, 0), dtype=X_groups[0].dtype)
        for i, X_group in enumerate(X_groups):
            assert(X_group.shape[0] == n_datas)
            X_group = X_group.reshape(n_datas, -1)
            if is_fit:
                group_dims.append( X_group.shape[1] )
                group_starts.append(0 if i == 0 else group_ends[i - 1])
                group_ends.append(group_starts[i] + group_dims[i])
            else:
                assert(X_group.shape[1] == group_dims[i])
            X = np.hstack((X, X_group))
        if is_fit:
            self.group_starts, self.group_ends, self.group_dims = group_starts, group_ends, group_dims
        return group_starts, group_ends, group_dims, X

    def fit_transform(self, X_groups_train, y_train, X_groups_test, y_test, stop_by_test=False, train_config=None):
        """
        fit until the accuracy converges in early_stop_rounds
        stop_by_test: (bool)
            When X_test, y_test is validation data that used for determine the opt_layer_id,
            use this option
        """
        if train_config is None:
            from ..config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir

        is_eval_test = "test" in train_config.phases
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        if is_eval_test and not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_train.shape={},y_train.shape={},X_groups_test.shape={},y_test.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape,
            [xt.shape for xt in X_groups_test] if is_eval_test else "no_test", y_test.shape if is_eval_test else "no_test"))

        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_train, True)
        if is_eval_test:
            self._check_look_indexs_cycle(X_groups_test, False)

        # check groups dimension
        group_starts, group_ends, group_dims, X_train = self._check_group_dims(X_groups_train, True)
        if is_eval_test:
            _, _, _, X_test = self._check_group_dims(X_groups_test, False)
        else:
            X_test = np.zeros((0, X_train.shape[1]))
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("group_starts={}".format(group_starts))
        LOGGER.info("group_ends={}".format(group_ends))
        LOGGER.info("X_train.shape={},X_test.shape={}".format(X_train.shape, X_test.shape))

        n_trains = X_groups_train[0].shape[0]
        n_tests = X_groups_test[0].shape[0] if is_eval_test else 0

        n_classes = self.n_classes
        assert n_classes == len(np.unique(y_train)), "n_classes({}) != len(unique(y)) {}".format(n_classes, np.unique(y_train))
        train_acc_list = []
        test_acc_list = []
        # X_train, y_train, X_test, y_test
        opt_datas = [None, None, None, None]
        try:
            # probability of each cascades's estimators
            X_proba_train = np.zeros((n_trains, n_classes*2), dtype=np.float32)
            X_proba_test = np.zeros((n_tests, n_classes*2), dtype=np.float32)
            # X_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            # X_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
            ECOC_output_matrix = np.zeros((n_trains, 6), dtype=np.float32)
            X_cur_train, X_cur_test = None, None
            layer_id = 0
            while 1:
                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                # Copy previous cascades's probability into current X_cur
                if layer_id == 0:
                    # first layer not have probability distribution
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)

                    X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                else:

                    X_cur_train = X_proba_train.copy()
                    X_cur_test = X_proba_test.copy()
                # Stack data that current layer needs in to X_cur
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
                for _i, i in enumerate(look_indexs):
                    X_cur_train = np.hstack((X_cur_train, X_train[:, group_starts[i]:group_ends[i]]))
                    X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}, X_cur_test.shape={}".format(
                    layer_id, look_indexs, X_cur_train.shape, X_cur_test.shape))
                # Fit on X_cur, predict to update X_proba
                y_train_proba_li = np.zeros((n_trains, n_classes))
                y_test_proba_li = np.zeros((n_tests, n_classes))
                # fix index
                # n_stratify = X_cur_train.shape[0]
                # if y_train is None:
                #     skf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                #     cv = [(t, v) for (t, v) in skf.split(len(y_train))]
                # else:
                #     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                #     cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_train)]
                for ei, est_config in enumerate(self.est_configs):

                    est = self._init_estimators(layer_id, ei)
                    # fit_trainsform
                    # split data
                    index_no_0 = [i for i, x in enumerate(y_train) if x == 1 or x == 2]
                    index_no_1 = [i for i, x in enumerate(y_train) if x == 0 or x == 2]
                    index_no_2 = [i for i, x in enumerate(y_train) if x == 0 or x == 1]
                    X_cur_train_0 = X_cur_train[index_no_0]
                    X_cur_train_1 = X_cur_train[index_no_1]
                    X_cur_train_2 = X_cur_train[index_no_2]
                    y_train_0 = y_train[index_no_0]
                    y_train_1 = y_train[index_no_1]
                    y_train_2 = y_train[index_no_2]
                    y_probas = list(np.zeros((1, n_trains, 3), dtype=np.float32))

                    test_sets = [("test", X_cur_test, y_test)] if n_tests > 0 else None
                    # y_probas = est.fit_transform(X_cur_train, y_train, y_train, layer_id,ei,
                    #         test_sets=test_sets, eval_metrics=self.eval_metrics,
                    #         keep_model_in_mem=train_config.keep_model_in_mem)
                    #

                    if ei == 0:
                        index_0 = [i for i, x in enumerate(y_train) if x == 0]
                        for i in range(len(index_no_0)):
                            y_train_0[i] -= 1

                        y_probas_0 = np.array(est.fit_transform(X_cur_train_0, y_train_0, y_train_0, layer_id, ei,
                                                     test_sets=test_sets, eval_metrics=self.eval_metrics,
                                                     keep_model_in_mem=train_config.keep_model_in_mem)).reshape((len(index_no_0),2))

                        add = np.zeros((len(index_no_0), 1), dtype=np.float32)
                        y_probas_0 = list(np.hstack((add, y_probas_0)))
                        y_probas[0][index_no_0] = y_probas_0

                        pro_0 = np.zeros((len(index_0), 3), dtype=np.float32)
                        pro_0 = list(pro_0)
                        y_probas[0][index_0] = pro_0
                        ecoc_y_pred = np.argmax(y_probas[0], axis=1)

                        for i in range(len(ecoc_y_pred)):
                            if ecoc_y_pred[i] == 2:
                                ecoc_y_pred[i] = -1
                        # print(ecoc_y_pred)
                        # print(y_probas[0][index_no_0])

                    if ei == 1:
                        for i in range(len(index_no_1)):
                            if y_train_1[i] == 2:
                                y_train_1[i] -= 1

                        y_probas_1 = np.array(est.fit_transform(X_cur_train_1, y_train_1, y_train_1, layer_id, ei,
                                                     test_sets=test_sets, eval_metrics=self.eval_metrics,
                                                     keep_model_in_mem=train_config.keep_model_in_mem)).reshape((len(index_no_1),2))
                        index_1 = [i for i, x in enumerate(y_train) if x == 1]
                        add = np.zeros((len(index_no_1), 3), dtype=np.float32)
                        for i in range(len(index_no_1)):
                            add[i][0] = y_probas_1[i][0]
                            add[i][2] = y_probas_1[i][1]
                        y_probas[0][index_no_1] = add
                        pro_1_a = np.zeros((len(index_1), 1), dtype=np.float32)
                        pro_1_b = np.ones((len(index_1), 1), dtype=np.float32)
                        pro_1_c = np.zeros((len(index_1), 1), dtype=np.float32)
                        pro_1 = np.hstack((pro_1_a, pro_1_b, pro_1_c))
                        pro_1 = list(pro_1)
                        y_probas[0][index_1] = pro_1
                        ecoc_y_pred = np.argmax(y_probas[0], axis=1)

                        for i in range(len(ecoc_y_pred)):
                            if ecoc_y_pred[i] == 2:
                                ecoc_y_pred[i] = -1
                            elif ecoc_y_pred[i] == 0:
                                ecoc_y_pred[i] = 1
                            elif ecoc_y_pred[i] == 1:
                                ecoc_y_pred[i] = 0
                        # print(ecoc_y_pred)

                        pro_1_new = np.zeros((len(index_1), 3), dtype=np.float32)
                        pro_1_new = list(pro_1_new)
                        y_probas[0][index_1] = pro_1_new
                        # print(y_probas[0][index_no_1])

                    if ei == 2:
                        y_probas_2 = np.array(est.fit_transform(X_cur_train_2, y_train_2, y_train_2, layer_id, ei,
                                                     test_sets=test_sets, eval_metrics=self.eval_metrics,
                                                     keep_model_in_mem=train_config.keep_model_in_mem)).reshape((len(index_no_2),2))
                        index_2 = [i for i, x in enumerate(y_train) if x == 2]
                        add = np.zeros((len(index_no_2), 1), dtype=np.float32)
                        y_probas_2 = list(np.hstack((y_probas_2, add)))
                        y_probas[0][index_no_2] = y_probas_2

                        pro_2 = np.zeros((len(index_2), 2), dtype=np.float32)
                        true_2 = np.ones((len(index_2), 1), dtype=np.float32)
                        pro_2 = np.hstack((pro_2, true_2))
                        pro_2 = list(pro_2)

                        y_probas[0][index_2] = pro_2
                        ecoc_y_pred = np.argmax(y_probas[0], axis=1)
                        for i in range(len(ecoc_y_pred)):
                            if ecoc_y_pred[i] == 2:
                                ecoc_y_pred[i] = 0
                            elif ecoc_y_pred[i] == 0:
                                ecoc_y_pred[i] = 1
                            elif ecoc_y_pred[i] == 1:
                                ecoc_y_pred[i] = -1

                        pro_2_new = np.zeros((len(index_2), 3), dtype=np.float32)
                        pro_2_new = list(pro_2_new)
                        y_probas[0][index_2] = pro_2_new
                        # print(y_probas[0][index_no_1])

                    if ei == 3:
                        y_train_3 = y_train.copy()
                        for i in range(n_trains):
                            if i in index_no_0:
                                y_train_3[i] = 1
                        y_probas_0 = np.array(est.fit_transform(X_cur_train, y_train_3, y_train_3, layer_id, ei,
                                                                test_sets=test_sets, eval_metrics=self.eval_metrics,
                                                                keep_model_in_mem=train_config.keep_model_in_mem)).reshape(
                            (n_trains, 2))
                        y_probas[0] = y_probas_0
                        ecoc_y_pred = np.argmax(y_probas[0], axis=1)
                        for i in range(len(ecoc_y_pred)):
                            if ecoc_y_pred[i] == 1:
                                ecoc_y_pred[i] = -1
                            if ecoc_y_pred[i] == 0:
                                ecoc_y_pred[i] = 1
                        # print(ecoc_y_pred)
                        # print(y_probas[0][index_no_0])

                    if ei == 4:
                        y_train_4 = y_train.copy()
                        for i in range(n_trains):
                            if i in index_no_1:
                                y_train_4[i] = 0


                        y_probas_1 = np.array(est.fit_transform(X_cur_train, y_train_4, y_train_4, layer_id, ei,
                                                                test_sets=test_sets, eval_metrics=self.eval_metrics,
                                                                keep_model_in_mem=train_config.keep_model_in_mem)).reshape(
                            (n_trains, 2))
                        y_probas[0] = y_probas_1
                        ecoc_y_pred = np.argmax(y_probas[0], axis=1)
                        for i in range(len(ecoc_y_pred)):
                            if ecoc_y_pred[i] == 0:
                                ecoc_y_pred[i] = -1
                            elif ecoc_y_pred[i] == 1:
                                ecoc_y_pred[i] = 1

                    if ei == 5:
                        y_train_5 = y_train.copy()
                        for i in range(n_trains):
                            if i in index_no_2:
                                y_train_5[i] = 0
                            else:
                                y_train_5[i] = 1
                        y_probas_2 = np.array(est.fit_transform(X_cur_train, y_train_5, y_train_5, layer_id, ei,
                                                                test_sets=test_sets, eval_metrics=self.eval_metrics,
                                                                keep_model_in_mem=train_config.keep_model_in_mem)).reshape(
                            (n_trains, 2))
                        y_probas[0] = y_probas_2
                        ecoc_y_pred = np.argmax(y_probas[0], axis=1)
                        for i in range(len(ecoc_y_pred)):
                            if ecoc_y_pred[i] == 0:
                                ecoc_y_pred[i] = -1
                            elif ecoc_y_pred[i] == 1:
                                ecoc_y_pred[i] = 1



                        # print(y_probas[0][index_no_1])

                    ECOC_output_matrix[:, ei] = ecoc_y_pred
                    # print(ECOC_output_matrix)

                        # train

                    # change class vector
                    # class_vector = y_probas[0].copy()
                    # y_true = y_train.reshape(-1)
                    # y_pred_temp = np.argmax(class_vector, axis=1)
                    # err_0_1, err_0_2, err_1_0, err_1_2, err_2_0, err_2_1 = [], [], [], [], [], []
                    # add_0_1, add_0_2, add_1_0, add_1_2, add_2_0, add_2_1 = 0,0,0,0,0,0
                    #
                    # med
                    # for i in range(len(y_true)):
                    #     if y_true[i] == 0 and y_true[i] != y_pred_temp[i]:
                    #         if y_pred_temp[i] == 1:
                    #             err_0_1.append(i)
                    #             diff = class_vector[i][1] - class_vector[i][0]
                    #             add_0_1 += diff
                    #         if y_pred_temp[i] == 2:
                    #             err_0_2.append(i)
                    #             diff = class_vector[i][2] - class_vector[i][0]
                    #             add_0_2 += diff
                    #     if y_true[i] == 1 and y_true[i] != y_pred_temp[i]:
                    #         if y_pred_temp[i] == 0:
                    #             err_1_0.append(i)
                    #             diff = class_vector[i][0] - class_vector[i][1]
                    #             add_1_0 += diff
                    #         if y_pred_temp[i] == 2:
                    #             err_1_2.append(i)
                    #             diff = class_vector[i][2] - class_vector[i][1]
                    #             add_1_2 += diff
                    #     if y_true[i] == 2 and y_true[i] != y_pred_temp[i]:
                    #         if y_pred_temp[i] == 0:
                    #             err_2_0.append(i)
                    #             diff = class_vector[i][0] - class_vector[i][2]
                    #             add_2_0 += diff
                    #         if y_pred_temp[i] == 1:
                    #             err_2_1.append(i)
                    #             diff = class_vector[i][1] - class_vector[i][2]
                    #             add_2_1 += diff
                    # add_0_1/=(len(err_0_1)+1)
                    # add_0_2/=(len(err_0_2)+1)
                    # add_1_0/=(len(err_1_0)+1)
                    # add_1_2/=(len(err_1_2)+1)
                    # add_2_0/=(len(err_2_0)+1)
                    # add_2_1/=(len(err_2_1)+1)
                    # for i in range(len(y_true)):
                    #     if i in err_0_1:
                    #         class_vector[i][0]+=add_0_1
                    #     if i in err_0_2:
                    #         class_vector[i][0]+=add_0_2
                    #     if i in err_1_0:
                    #         class_vector[i][1]+=add_1_0
                    #     if i in err_1_2:
                    #         class_vector[i][1] += add_1_2
                    #     if i in err_2_0:
                    #         class_vector[i][2]+=add_2_0
                    #     if i in err_2_1:
                    #         class_vector[i][2]+=add_2_1

                    # # # # # # # # reset
                    # for i in range(len(y_true)):
                    #     if y_true[i] == 0 and y_pred_temp[i] != y_true[i]:
                    #         class_vector[i] = [1, 0, 0]
                    #     if y_true[i] == 1 and y_pred_temp[i] != y_true[i]:
                    #         class_vector[i] = [0, 1, 0]
                    #     if y_true[i] == 2 and y_pred_temp[i] != y_true[i]:
                    #         class_vector[i] = [0, 0, 1]


                    # X_proba_train[:, ei * n_classes: ei * n_classes + n_classes] = class_vector

                    # X_proba_train[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[0]

                    # y_train_proba_li += y_probas[0]
                    # test
                    if n_tests > 0:
                        X_proba_test[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[1]
                        y_test_proba_li += y_probas[1]
                    if train_config.keep_model_in_mem:
                        self._set_estimator(layer_id, ei, est)

                # print(ECOC_output_matrix)
                distance_pro1, distance_pro2 = self.get_distance_pro(ECOC_output_matrix, y_train)
                X_proba_train_temp = distance_pro1 + distance_pro2

                class_vector1 = distance_pro1.copy()
                class_vector2 = distance_pro2.copy()
                y_true = y_train.reshape(-1)
                y_pred_temp = np.argmax(class_vector1, axis=1)

                # for i in range(len(y_true)):
                #     if y_pred_temp[i] == y_true[i] == 0 and abs(class_vector1[i][0] - max(class_vector1[i][1], class_vector1[i][2])) <= 0.2:
                #         class_vector1[i] = [1, 0, 0]
                #     elif y_pred_temp[i] == y_true[i] == 1 and abs(class_vector1[i][1] - max(class_vector1[i][0], class_vector1[i][2])) <= 0.2:
                #         class_vector1[i] = [0, 1, 0]
                #     elif y_pred_temp[i] == y_true[i] == 2 and abs(class_vector1[i][2] - max(class_vector1[i][1], class_vector1[i][0])) <= 0.2:
                #         class_vector1[i] = [0, 0, 1]
                # for i in range(len(y_true)):
                #     if y_pred_temp[i] == y_true[i] == 0 and abs(class_vector2[i][0] - max(class_vector2[i][1],class_vector2[i][2])) <= 0.2:
                #         class_vector2[i] = [1, 0, 0]
                #     elif y_pred_temp[i] == y_true[i] == 1 and abs(class_vector2[i][1] - max(class_vector2[i][0],class_vector2[i][2])) <= 0.2:
                #         class_vector2[i] = [0, 1, 0]
                #     elif y_pred_temp[i] == y_true[i] == 2 and abs(class_vector2[i][2] - max(class_vector2[i][1],class_vector2[i][0])) <= 0.2:
                #         class_vector2[i] = [0, 0, 1]

                new_class_vector1 = self.reset_vector(class_vector1, y_train)
                new_class_vector2 = self.reset_vector(class_vector2, y_train)
                X_proba_train = np.hstack((new_class_vector1, new_class_vector2))
                # print(X_proba_train)
                # X_proba_train = np.hstack((class_vector1, class_vector2))

                # y_train_proba_li /= len(self.est_configs)
                # X_proba_train = y_train_proba_li
                train_avg_acc = calc_accuracy(y_train, np.argmax(X_proba_train_temp, axis=1),
                                              'layer_{} - train.classifier_average'.format(layer_id))
                # train_avg_acc = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1), 'layer_{} - train.classifier_average'.format(layer_id))
                train_acc_list.append(train_avg_acc)

                # print csv
                # y_true = y_train.reshape(-1)
                # y_pred = np.argmax(y_train_proba_li, axis=1)
                # err = []
                # prob = []
                # for i in range(len(y_true)):
                #     if y_true[i] != y_pred[i]:
                #         err.append(i)
                #         prob.append(y_train_proba_li[i])
                # dataname = str(layer_id) + '.csv'
                # result = pd.DataFrame({'y_true': y_true[err], 'y_pred': y_pred[err], 'y_prob': prob, 'err':err})
                # result.to_csv('../result/' + dataname)
                if n_tests > 0:
                    y_test_proba_li /= len(self.est_configs)
                    test_avg_acc = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1), 'layer_{} - test.classifier_average'.format(layer_id))
                    test_acc_list.append(test_avg_acc)
                else:
                    test_acc_list.append(0.0)

                opt_layer_id = get_opt_layer_id(test_acc_list if stop_by_test else train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [X_proba_train, y_train, X_proba_test if n_tests > 0 else None, y_test]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                # if self.early_stopping_rounds > 0 and layer_id == 3:
                    # log and save final result (opt layer)
                    # opt_layer_id = 3
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                        opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
                    if data_save_dir is not None:
                        self.save_data(data_save_dir, opt_layer_id, *opt_datas)

                    # remove unused model
                    if train_config.keep_model_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            for ei, est_config in enumerate(self.est_configs):
                                self._set_estimator(li, ei, None)
                    self.opt_layer_num = opt_layer_id + 1
                    return opt_layer_id, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(data_save_dir, layer_id, *opt_datas)
                # inc layer_id
                layer_id += 1

            LOGGER.info("[Result][Reach Max Layer] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
            if data_save_dir is not None:
                self.save_data(data_save_dir, self.max_layers - 1, *opt_datas)
            self.opt_layer_num = self.max_layers
            return self.max_layers, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
        except KeyboardInterrupt:
            pass

    def get_distance_pro(self, ecoc_matrix, y_train):
        distance_pro_temp_1 = np.zeros((np.shape(ecoc_matrix)[0], 3), dtype=np.float32)
        distance_pro_temp_2 = np.zeros((np.shape(ecoc_matrix)[0], 3), dtype=np.float32)
        # distance_pro = np.zeros((np.shape(ecoc_matrix)[0], np.shape(ecoc_matrix)[1]), dtype=np.float32)
        matrix_1 = [[0, 1, 1], [1, 0, -1], [-1, -1, 0]]
        matrix_2 = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
        ecoc_matrix_1 = ecoc_matrix[:, :3]
        ecoc_matrix_2 = ecoc_matrix[:, 3:]
        for i, j in enumerate(ecoc_matrix_1):
            for m, n in enumerate(matrix_1):
                j = np.array(j).reshape(1, -1)
                n = np.array(n).reshape(1, -1)
                # distance = np.sum(((1 - np.sign(j * n)) / 2))
                distance = np.sqrt(np.sum(np.power(j - n, 2)))
                distance_pro_temp_1[i][m] = distance
            nums = np.sum(distance_pro_temp_1[i])
            distance_pro_temp_1[i][0] /= nums
            distance_pro_temp_1[i][1] /= nums
            distance_pro_temp_1[i][2] /= nums
            sort = np.argsort(distance_pro_temp_1[i])
            min_pro = sort[0]
            med_pro = sort[1]
            max_pro = sort[2]
            if min_pro == y_train[i] and distance_pro_temp_1[i][max_pro] == distance_pro_temp_1[i][med_pro]:
                distance_pro_temp_1[i][min_pro] = 0.7
                distance_pro_temp_1[i][max_pro] = 0.15
                distance_pro_temp_1[i][med_pro] = 0.15
            else:
                temp = distance_pro_temp_1[i][min_pro]
                distance_pro_temp_1[i][min_pro] = distance_pro_temp_1[i][max_pro]+0.1
                distance_pro_temp_1[i][max_pro] = temp

        for i, j in enumerate(ecoc_matrix_2):
            for m, n in enumerate(matrix_2):
                j = np.array(j).reshape(1, -1)
                n = np.array(n).reshape(1, -1)
                # distance = np.sum(((1 - np.sign(j * n)) / 2))
                distance = np.sqrt(np.sum(np.power(j - n, 2)))
                distance_pro_temp_2[i][m] = distance
            nums = np.sum(distance_pro_temp_2[i])
            distance_pro_temp_2[i][0] /= nums
            distance_pro_temp_2[i][1] /= nums
            distance_pro_temp_2[i][2] /= nums
            sort = np.argsort(distance_pro_temp_2[i])
            min_pro = sort[0]
            med_pro = sort[1]
            max_pro = sort[2]
            if min_pro == y_train[i] and distance_pro_temp_2[i][max_pro] == distance_pro_temp_2[i][med_pro]:
                distance_pro_temp_2[i][min_pro] = 0.7
                distance_pro_temp_2[i][max_pro] = 0.15
                distance_pro_temp_2[i][med_pro] = 0.15
            else:
                temp = distance_pro_temp_2[i][min_pro]
                distance_pro_temp_2[i][min_pro] = distance_pro_temp_2[i][max_pro]+0.1
                distance_pro_temp_2[i][max_pro] = temp

        return distance_pro_temp_1, distance_pro_temp_2

    def reset_vector(self, class_vector, y_train):
        y_true = y_train.reshape(-1)
        y_pred_temp = np.argmax(class_vector, axis=1)
        err_0_1, err_0_2, err_1_0, err_1_2, err_2_0, err_2_1 = [], [], [], [], [], []
        add_0_1, add_0_2, add_1_0, add_1_2, add_2_0, add_2_1 = 0, 0, 0, 0, 0, 0

        for i in range(len(y_true)):
            if y_true[i] == 0 and y_pred_temp[i] != y_true[i]:
                class_vector[i] = [1, 0, 0]
            if y_true[i] == 1 and y_pred_temp[i] != y_true[i]:
                class_vector[i] = [0, 1, 0]
            if y_true[i] == 2 and y_pred_temp[i] != y_true[i]:
            # if y_true[i] == 2:
                class_vector[i] = [0, 0, 1]

        # for i in range(len(y_true)):
        #     if y_true[i] == 0 and y_true[i] != y_pred_temp[i]:
        #         if y_pred_temp[i] == 1:
        #             err_0_1.append(i)
        #             diff = class_vector[i][1] - class_vector[i][0]
        #             add_0_1 += diff
        #         if y_pred_temp[i] == 2:
        #             err_0_2.append(i)
        #             diff = class_vector[i][2] - class_vector[i][0]
        #             add_0_2 += diff
        #     if y_true[i] == 1 and y_true[i] != y_pred_temp[i]:
        #         if y_pred_temp[i] == 0:
        #             err_1_0.append(i)
        #             diff = class_vector[i][0] - class_vector[i][1]
        #             add_1_0 += diff
        #         if y_pred_temp[i] == 2:
        #             err_1_2.append(i)
        #             diff = class_vector[i][2] - class_vector[i][1]
        #             add_1_2 += diff
        #     if y_true[i] == 2 and y_true[i] != y_pred_temp[i]:
        #         if y_pred_temp[i] == 0:
        #             err_2_0.append(i)
        #             diff = class_vector[i][0] - class_vector[i][2]
        #             add_2_0 += diff
        #         if y_pred_temp[i] == 1:
        #             err_2_1.append(i)
        #             diff = class_vector[i][1] - class_vector[i][2]
        #             add_2_1 += diff
        # add_0_1 /= (len(err_0_1)+1)
        # add_0_2 /= (len(err_0_2)+1)
        # add_1_0 /= (len(err_1_0)+1)
        # add_1_2 /= (len(err_1_2)+1)
        # add_2_0 /= (len(err_2_0)+1)
        # add_2_1 /= (len(err_2_1)+1)
        # for i in range(len(y_true)):
        #     if i in err_0_1:
        #         class_vector[i][0] += add_0_1
        #     if i in err_0_2:
        #         class_vector[i][0] += add_0_2
        #     if i in err_1_0:
        #         class_vector[i][1] += add_1_0
        #     if i in err_1_2:
        #         class_vector[i][1] += add_1_2
        #     if i in err_2_0:
        #         class_vector[i][2] += add_2_0
        #     if i in err_2_1:
        #         class_vector[i][2] += add_2_1

        return class_vector

    def transform(self, X_groups_test):
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_test.shape={}".format([xt.shape for xt in X_groups_test]))
        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_test, False)
        # check group_dims
        group_starts, group_ends, group_dims, X_test = self._check_group_dims(X_groups_test, False)
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("X_test.shape={}".format(X_test.shape))

        n_tests = X_groups_test[0].shape[0]
        n_classes = self.n_classes
        # probability of each cascades's estimators
        # X_proba_test = np.zeros((X_test.shape[0], n_classes * self.n_estimators_1), dtype=np.float32)
        X_proba_test = np.zeros((n_tests, n_classes * 2), dtype=np.float32)
        ECOC_TEST_MATRIX = np.zeros((n_tests, n_classes*2), dtype=np.float32)
        Every_forest_predict = np.zeros((n_tests, n_classes * 2), dtype=np.float32)
        X_cur_test = None
        for layer_id in range(self.opt_layer_num):

            y_test_proba_li = np.zeros((n_tests, n_classes))
            # Copy previous cascades's probability into current X_cur
            if layer_id == 0:
                # first layer not have probability distribution
                X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
            else:
                X_cur_test = X_proba_test.copy()
            # Stack data that current layer needs in to X_cur
            look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
            for _i, i in enumerate(look_indexs):
                X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
            LOGGER.info("[layer={}] look_indexs={}, X_cur_test.shape={}".format(
                layer_id, look_indexs, X_cur_test.shape))
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                if est is None:
                    raise ValueError("model (li={}, ei={}) not present, maybe you should set keep_model_in_mem to True".format(
                        layer_id, ei))
                y_probas, predict_tree = est.predict_proba(X_cur_test)
                # 生成每棵树的结果
                # data_tree = DataFrame(predict_tree).T
                # name = str(layer_id)+ str(ei)+'.csv'
                # data_tree.to_csv('../CASME/' + name, mode='a', header=False )

                if ei == 0:
                    a = np.zeros((n_tests, 1), dtype=np.float32)
                    y_probas = np.hstack((a, y_probas))
                    # print(y_probas )
                    ecoc_ytest_pred = np.argmax(y_probas, axis=1)
                    Every_forest_predict[:, ei] = ecoc_ytest_pred.copy()
                    for i in range(len(ecoc_ytest_pred)):
                        if ecoc_ytest_pred[i] == 2:
                            ecoc_ytest_pred[i] = -1
                    # print("ei0")
                    # print(ecoc_ytest_pred)

                if ei == 1:
                    a = np.zeros((n_tests, 3), dtype=np.float32)
                    for i in range(n_tests):
                        a[i][0] = y_probas[i][0]
                        a[i][2] = y_probas[i][1]
                    y_probas = a
                    # print(y_probas)
                    ecoc_ytest_pred = np.argmax(y_probas, axis=1)
                    Every_forest_predict[:, ei] = ecoc_ytest_pred.copy()
                    for i in range(len(ecoc_ytest_pred)):
                        if ecoc_ytest_pred[i] == 2:
                            ecoc_ytest_pred[i] = -1
                        elif ecoc_ytest_pred[i] == 0:
                            ecoc_ytest_pred[i] = 1
                    # print("ei1")
                    # print(ecoc_ytest_pred)

                if ei == 2:
                    a = np.zeros((n_tests, 1), dtype=np.float32)
                    y_probas = np.hstack((y_probas, a))
                    # print(y_probas)
                    ecoc_ytest_pred = np.argmax(y_probas, axis=1)
                    Every_forest_predict[:, ei] = ecoc_ytest_pred.copy()
                    for i in range(len(ecoc_ytest_pred)):
                        if ecoc_ytest_pred[i] == 0:
                            ecoc_ytest_pred[i] = 1
                        elif ecoc_ytest_pred[i] == 1:
                            ecoc_ytest_pred[i] = -1
                    # print("ei2")
                    # print(ecoc_ytest_pred)

                if ei == 3:
                    ecoc_ytest_pred = np.argmax(y_probas, axis=1)
                    Every_forest_predict[:, ei] = ecoc_ytest_pred.copy()
                    for i in range(len(ecoc_ytest_pred)):
                        if ecoc_ytest_pred[i] == 0:
                            ecoc_ytest_pred[i] = 1
                        else:
                            ecoc_ytest_pred[i] = -1
                    # print("ei3")
                    # print(ecoc_ytest_pred)

                if ei == 4:
                    ecoc_ytest_pred = np.argmax(y_probas, axis=1)
                    Every_forest_predict[:, ei] = ecoc_ytest_pred.copy()
                    for i in range(len(ecoc_ytest_pred)):
                        if ecoc_ytest_pred[i] == 1:
                            ecoc_ytest_pred[i] = 1
                        elif ecoc_ytest_pred[i] == 0:
                            ecoc_ytest_pred[i] = -1
                    # print("ei4")
                    # print(ecoc_ytest_pred)

                if ei == 5:
                    ecoc_ytest_pred = np.argmax(y_probas, axis=1)
                    Every_forest_predict[:, ei] = ecoc_ytest_pred.copy()
                    for i in range(len(ecoc_ytest_pred)):
                        if ecoc_ytest_pred[i] == 0:
                            ecoc_ytest_pred[i] = -1
                        elif ecoc_ytest_pred[i] == 1:
                            ecoc_ytest_pred[i] = 1
                    # print("ei5")
                    # print(ecoc_ytest_pred)

                ECOC_TEST_MATRIX[:, ei] = ecoc_ytest_pred
                # y_test_proba_li += y_probas
            # X_proba_test[:, ei * n_classes:ei * n_classes + n_classes] = y_probas
            distance_pro = self.get_test_distance_pro(ECOC_TEST_MATRIX)
            X_proba_test = distance_pro
            # y_test_proba_li /= len(self.est_configs)
            # X_proba_test = y_test_proba_li
            # print(X_proba_test)
            # print(y_test_proba_li)
        return X_proba_test

    def predict_proba(self, X):
        # n x (n_est*n_classes)
        y_proba = self.transform(X)
        # n x n_est x n_classes
        # y_proba = y_proba.reshape((y_proba.shape[0], self.n_estimators_1, self.n_classes))
        y_proba = y_proba.reshape((y_proba.shape[0], 2, self.n_classes))
        y_proba = y_proba.mean(axis=1)
        return y_proba

    def get_test_distance_pro(self, ecoc_matrix):
        distance_pro_temp1 = np.zeros((np.shape(ecoc_matrix)[0], 3), dtype=np.float32)
        distance_pro_temp2 = np.zeros((np.shape(ecoc_matrix)[0], 3), dtype=np.float32)
        distance_test_pro1 = np.zeros((np.shape(ecoc_matrix)[0], 3), dtype=np.float32)
        distance_test_pro2 = np.zeros((np.shape(ecoc_matrix)[0], 3), dtype=np.float32)
        matrix = [[0, 1, 1], [1, 0, -1], [-1, -1, 0]]
        matrix_2 = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
        ecoc_matrix_1 = ecoc_matrix[:, :3]
        ecoc_matrix_2 = ecoc_matrix[:, 3:]
        for i, j in enumerate(ecoc_matrix_1):
            for m, n in enumerate(matrix):
                j = np.array(j).reshape(1, -1)
                n = np.array(n).reshape(1, -1)
                # distance = np.sum(((1 - np.sign(j * n)) / 2))
                distance = np.sqrt(np.sum(np.power(j - n, 2)))
                distance_pro_temp1[i][m] = distance
            nums = np.sum(distance_pro_temp1[i])
            distance_pro_temp1[i][0] /= nums
            distance_pro_temp1[i][1] /= nums
            distance_pro_temp1[i][2] /= nums

            sort = np.argsort(distance_pro_temp1[i])
            min_pro = sort[0]
            med_pro = sort[1]
            max_pro = sort[2]
            distance_test_pro1[i][min_pro] = distance_pro_temp1[i][max_pro]+0.1
            distance_test_pro1[i][med_pro] = distance_pro_temp1[i][med_pro]
            distance_test_pro1[i][max_pro] = distance_pro_temp1[i][min_pro]

        # y_pro1 = np.argmin(distance_pro_temp1, axis=1)

        for i, j in enumerate(ecoc_matrix_2):
            for m, n in enumerate(matrix_2):
                j = np.array(j).reshape(1, -1)
                n = np.array(n).reshape(1, -1)
                # distance = np.sum(((1 - np.sign(j * n)) / 2))
                distance = np.sqrt(np.sum(np.power(j - n, 2)))
                distance_pro_temp2[i][m] = distance
            nums = np.sum(distance_pro_temp2[i])
            distance_pro_temp2[i][0] /= nums
            distance_pro_temp2[i][1] /= nums
            distance_pro_temp2[i][2] /= nums

            sort = np.argsort(distance_pro_temp2[i])
            min_pro = sort[0]
            med_pro = sort[1]
            max_pro = sort[2]
            distance_test_pro2[i][min_pro] = distance_pro_temp2[i][max_pro]+0.1
            distance_test_pro2[i][med_pro] = distance_pro_temp2[i][med_pro]
            distance_test_pro2[i][max_pro] = distance_pro_temp2[i][min_pro]

        # y_pro2 = np.argmin(distance_pro_temp2, axis=1)

        # for i in range(len(y_pro1)):
        #     if y_pro1[i] == 0:
        #         distance_test_pro1[i] = [0.6, 0.2, 0.2]
        #     elif y_pro1[i] == 1:
        #         distance_test_pro1[i] = [0.2, 0.6, 0.2]
        #     elif y_pro1[i] == 2:
        #         distance_test_pro1[i] = [0.2, 0.2, 0.6]
        #
        # for i in range(len(y_pro2)):
        #     if y_pro2[i] == 0:
        #         distance_test_pro2[i] = [0.6, 0.2, 0.2]
        #     elif y_pro2[i] == 1:
        #         distance_test_pro2[i] = [0.2, 0.6, 0.2]
        #     elif y_pro2[i] == 2:
        #         distance_test_pro2[i] = [0.2, 0.2, 0.6]
        distance_pro = np.hstack((distance_test_pro1, distance_test_pro2))
        # print(distance_pro)
        return distance_pro



    def save_data(self, data_save_dir, layer_id, X_train, y_train, X_test, y_test):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train} if pi == 0 else {"X": X_test, "y": y_test}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


    def auto_data(self, X, y,source_pictre):
        LOGGER.info('plot..')
        test_images = X.reshape((X.shape[0], -1))
        test_images = test_images.reshape(1, X.shape[0], 28, 28, 3).astype(np.uint8)
        # print(np.shape(test_images))
        source = source_pictre.reshape((source_pictre.shape[0], -1))

        source_pictre_real = source_pictre.reshape(1, source.shape[0], 170, 140, 3).astype(np.uint8)
        for layer_id in range(self.opt_layer_num):
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                # est.auto_kfold(X, y, layer_id, ei)
# according to layer to show
                result = est.auto_kfold(X, y, layer_id, ei)
                if ei == 0:
                    datas = np.vstack((test_images, result))
                else:
                    datas = np.vstack((datas, result))

            rheads = ["ApexFrame", "Feature", "Classifier1", "Classifier2", "Classifier3", "Classifier4", "Classifier5", "Classifier6"]
            # fig = self.plot_micro(rheads, datas, source_pictre_real)
            fig = self.plot_micro_renew(rheads, datas, source_pictre_real)

            fig.set_size_inches(3.5, 5.5)
            # plt.savefig('./'+ str(layer_id)+'.png')
            plt.show()


# according to layer add
#                 result = est.auto_kfold(X, y, layer_id, ei)
#                 # print(np.shape(result))
#
#                 if layer_id == 0:
#                     if ei == 0:
#                         result_0 = result.copy()
#                     elif ei == 1:
#                         result_1 = result.copy()
#                     elif ei == 2:
#                         result_2 = result.copy()
#                     elif ei == 3:
#                         result_3 = result.copy()
#                     elif ei == 4:
#                         result_4 = result.copy()
#                     elif ei == 5:
#                         result_5 = result.copy()
#                 else:
#                     if ei == 0:
#                         result_0 += result
#                     elif ei == 1:
#                         result_1 += result
#                     elif ei == 2:
#                         result_2 += result
#                     elif ei == 3:
#                         result_3 += result
#                     elif ei == 4:
#                         result_4 += result
#                     elif ei == 5:
#                         result_5 += result
#
#         result_0 /= self.opt_layer_num
#         result_1 /= self.opt_layer_num
#         result_2 /= self.opt_layer_num
#         result_3 /= self.opt_layer_num
#         result_4 /= self.opt_layer_num
#         result_5 /= self.opt_layer_num
#
#         # # # plot
#         rheads = ["origin", "classifier0", "classifier1", "classifier2", "classifier3", "classifier4", "classifier5"]
#         datas = np.vstack((test_images, result_0, result_1, result_2, result_3, result_4, result_5))
#
#         fig = self.plot_micro(rheads, datas)
#         fig.set_size_inches(6, 15)
#         plt.show()

   # according to  different value
#                 result = est.auto_kfold(X, y, layer_id, ei)
#             # print(np.shape(result))
#
#                 if layer_id == 0:
#                     if ei == 0:
#                         result_0 = result.copy()
#                     elif ei == 1:
#                         result_1 = result.copy()
#                     elif ei == 2:
#                         result_2 = result.copy()
#                     elif ei == 3:
#                         result_3 = result.copy()
#                     elif ei == 4:
#                         result_4 = result.copy()
#                     elif ei == 5:
#                         result_5 = result.copy()
#                 else:
#                     if ei == 0:
#                         result_0 -= result
#                     elif ei == 1:
#                         result_1 -= result
#                     elif ei == 2:
#                         result_2 -= result
#                     elif ei == 3:
#                         result_3 -= result
#                     elif ei == 4:
#                         result_4 -= result
#                     elif ei == 5:
#                         result_5 -= result
#
#
#
# # # plot
#         rheads = ["origin", "classifier0", "classifier1", "classifier2", "classifier3", "classifier4", "classifier5"]
#         datas = np.vstack((test_images, result_0, result_1, result_2, result_3, result_4, result_5))
#         fig = self.plot_micro(rheads, datas)
#         fig.set_size_inches(6, 15)
#         plt.savefig('different.png')
#         plt.show()

    def plot_micro(self, rheads, datas,source_picture_real):
        """
        datas: ndarray
            shape = [n_rows, 10, 3072]
        """
        n_rows = len(rheads)
        n_cols = len(datas[0])
        fig = plt.figure()
        for r in range(n_rows):
            fig.add_subplot(n_rows * 2, n_cols, r * 2 * n_cols + 0 + 1)
            plt.text(0, 0, rheads[r])
            plt.axis("off")
            for c in range(n_cols):
                if r ==0:
                    fig.add_subplot(n_rows * 2, n_cols, (r * 2 + 1) * n_cols + c + 1)
                    B, G, R = cv2.split(source_picture_real[r][c])
                    img = cv2.merge([R, G, B])
                    img = np.array(img).astype(np.uint8)
                    plt.imshow(img)
                    plt.axis("off")
                else:
                    a = r-1
                    fig.add_subplot(n_rows * 2, n_cols, (r * 2 + 1) * n_cols + c + 1)
                    B, G, R = cv2.split(datas[a][c])
                    img = cv2.merge([R, G, B])
                    img = np.array(img).astype(np.uint8)
                    plt.imshow(img)
                    plt.axis("off")
        return fig

    def plot_micro_renew(self, rheads, datas,source_picture_real):
        """
        datas: ndarray
            shape = [n_rows, 10, 3072]
        """
        n_rows = len(rheads)
        n_cols = len(datas[0])
        fig = plt.figure()
        for r in range(n_rows):
            fig.add_subplot(n_rows, 4, r * 4 + 0 + 1)
            plt.text(0.25, 0.6, rheads[r], size=7.5)
            plt.axis("off")
            for c in range(n_cols):
                if r == 0:
                    fig.add_subplot(n_rows, 4, r * 4 + c + 2)
                    B, G, R = cv2.split(source_picture_real[r][c])
                    img = cv2.merge([R, G, B])
                    img = np.array(img).astype(np.uint8)
                    plt.imshow(img)
                    plt.axis("off")
                else:
                    a = r-1
                    fig.add_subplot(n_rows, 4, r * 4 + c + 2)
                    B, G, R = cv2.split(datas[a][c])
                    img = cv2.merge([R, G, B])
                    img = np.array(img).astype(np.uint8)
                    plt.imshow(img)
                    name = str(c)
                    matplotlib.image.imsave(rheads[r]+name+'.png', img)
                    plt.axis("off")
        return fig








