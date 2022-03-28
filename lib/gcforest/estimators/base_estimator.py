# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import os, os.path as osp
import numpy as np
import cv2

import matplotlib.pyplot as plt
from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path
import matplotlib.image as mping
LOGGER = get_logger("gcforest.estimators.base_estimator")

def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

class BaseClassifierWrapper(object):
    def __init__(self, name, est_class, est_args):
        """
        name: str)
            Used for debug and as the filename this model may be saved in the disk
        """
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.cache_suffix = ".pkl"
        self.est = None

    def _init_estimator(self):
        """
        You can re-implement this function when inherient this class
        """
        est = self.est_class(**self.est_args)
        return est

    def fit(self, X, y, cache_dir=None):
        """
        cache_dir(str): 
            if not None
                then if there is something in cache_dir, dont have fit the thing all over again
                otherwise, fit it and save to model cache 
        """
        LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if self._is_cache_exists(cache_path):
            LOGGER.info("Find estimator from {} . skip process".format(cache_path))
            return
        est = self._init_estimator()
        self._fit(est, X, y)
        if cache_path is not None:
            # saved in disk
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path); 
            self._save_model_to_disk(est, cache_path)
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
        return y_proba

    def auto_baseestimator(self, X, y, layer_id,ei):
        clf = self.est
        x_test = X.reshape((X.shape[0], -1))
        test_images = np.zeros((3, 28 * 28*3))
        for label in range(3):
            index = np.where(y == label)[0][0]
            test_images[label] = x_test[index]
        # results = np.zeros((3, 28 * 28, 3))
        # print(test_images)
        if layer_id == 0:
            x_encode = clf.encode(test_images)
            x_decode = clf.decode(x_encode)
            test_images = test_images.reshape(1, 3, 28, 28, 3).astype(np.uint8)
            # test_images = test_images.reshape(1, 3, 28, 28, 3)
            result = x_decode.reshape(3, 28 * 28, 3)
            results = result.reshape(1, 3, 28, 28, 3).astype(np.uint8)
            # results = result.reshape(1, 3, 28, 28, 3)

        else:
            x_make = np.zeros((3, 6))
            new_test_images = np.hstack((x_make, test_images))
            x_encode = clf.encode(new_test_images)
            x_decode = clf.decode(x_encode)
            new_x_decode = np.delete(x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            test_images = test_images.reshape(1, 3, 28, 28, 3).astype(np.uint8)
            # test_images = test_images.reshape(1, 3, 28, 28, 3)
            result = new_x_decode.reshape(3, 28 * 28, 3)
            results = result.reshape(1, 3, 28, 28, 3).astype(np.uint8)
            # results = result.reshape(1, 3, 28, 28, 3)
        name = "layer{},ei{}".format(layer_id, ei)
        # rheads = ["origin", "horizontal", "vertical", "os", name, "horizontal", "vertical", "os"]
        rheads = ["origin", name]
        datas = np.vstack((test_images, results))
        # fig = self.plot_micro_test(rheads, datas)
        fig = self.plot_micro(rheads, datas)
        plt.show()

    def auto_baseestimator_test(self, X, y, layer_id, ei):
        clf = self.est
        x_test = X.reshape((X.shape[0], -1))
        test_images = x_test
        a =X.shape[0]

        if layer_id == 0:
            x_encode = clf.encode(test_images)
            x_decode = clf.decode(x_encode)
            test_images = test_images.reshape(1, a, 28, 28, 3).astype(np.uint8)
            # test_images = test_images.reshape(1, a, 28, 28, 3)
            result = x_decode.reshape(a, 28 * 28, 3)
            results = result.reshape(1, a, 28, 28, 3).astype(np.uint8)
            # results = result.reshape(1, a, 28, 28, 3)

        else:
            x_make = np.zeros((a, 6))
            new_test_images = np.hstack((x_make, test_images))
            x_encode = clf.encode(new_test_images)
            x_decode = clf.decode(x_encode)
            new_x_decode = np.delete(x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)
            new_x_decode = np.delete(new_x_decode, 0, axis=1)

            test_images = test_images.reshape(1, a, 28, 28, 3).astype(np.uint8)
            result = new_x_decode.reshape(a, 28 * 28, 3)
            results = result.reshape(1, a, 28, 28, 3).astype(np.uint8)

        name = "layer{},ei{}".format(layer_id, ei)
        # rheads = ["origin", "horizontal", "vertical", "os", name, "horizontal", "vertical", "os"]
        rheads = ["origin", name]
        datas = np.vstack((test_images, results))
        # fig = self.plot_micro_test(rheads, datas)
        # fig = self.plot_micro(rheads, datas)
        # plt.show()
        # print("in")
        # print(results)
        return results


    def plot_micro(self, rheads, datas):
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
                fig.add_subplot(n_rows * 2, n_cols, (r * 2 + 1) * n_cols + c + 1)
                B, G, R = cv2.split(datas[r][c])
                img = cv2.merge([R, G, B])
                img = np.array(img).astype(np.uint8)
                plt.imshow(img)
                    # plt.imshow(datas[r][c]/255)
        return fig

    def plot_micro_test(self,rheads, datas):
        """
        datas: ndarray
            shape = [n_rows, 10, 3072]
        """
        # rheads = ["origin", "horizontal", "vertical", "os", "123", "horizontal", "vertical", "os"]
        # n_rows = len(rheads)
        n_rows = len(datas)
        n_cols = len(datas[0])+1
        t_row = n_rows*4
        n = 1
        head_n = 0
        fig = plt.figure(figsize=[3, 8])
        for i in range(len(rheads)):
            plt.subplot(t_row, n_cols, i*4+1)
            plt.axis("off")
            plt.text(0, 0.5, rheads[i])
        for r in range(n_rows):
            n += 1
            for c in range(n_cols-1):
                plt.subplot(t_row, n_cols, n)
                plt.axis("off")
                plt.imshow(datas[r][c])
                channels = datas[r][c].shape[-1]
                temp_n = n
                for channel in range(channels):
                    temp_n += n_cols
                    plt.subplot( t_row,n_cols, temp_n)
                    plt.axis("off")
                    a = datas[r][c][:, :, channel]
                    plt.imshow(datas[r][c][:, :, channel])
                n += 1
            n += 3 * n_cols
        return fig



    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _is_cache_exists(self, cache_path):
        return cache_path is not None and osp.exists(cache_path)

    def _batch_predict_proba(self, est, X, batch_size):
        LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            LOGGER.info("[progress][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            y_cur = self._predict_proba(est, X[j:j+batch_size])
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
            y_pred_proba[j:j+batch_size,:] = y_cur
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _load_model_from_disk(self, cache_path):
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        raise NotImplementedError()

    def _default_predict_batch_size(self, est, X):
        """
        You can re-implement this function when inherient this class 

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba without baches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _fit(self, est, X, y):
        est.fit(X, y)


    def _predict_proba(self, est, X):

        return est.predict_proba(X)

    def _predict_from_tree(self, X):
        est = self.est
        if hasattr(est, 'estimators_'):
            result = []
            for estimator in est.estimators_:
                result.append(estimator.predict(X))
            return result
        else:
            print("model hasn't trained! ")
            return None



