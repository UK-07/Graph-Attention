import os
import shutil

import torch

import numpy as np
from collections import defaultdict

def removeIsolated(suffix = '.f.npy'):
    '''
    remove isolated points which have no neighbors.
    return number of nodes and categories.
    '''
    src_folder = '/path/to/train_feature'
    class_folders = os.listdir(src_folder)
    node_num = 0
    class_num = 0
    for c, folder in enumerate(class_folders):
        path = os.path.join(src_folder, folder)
        item_num = len(filter(lambda f: f.endswith(suffix), os.listdir(path)))
        if item_num > 1:
            node_num += item_num
            class_num += 1
        else:
            print "remove class-{}".format(c)
            shutil.rmtree(path)

    print "node num.:", node_num
    print "class num.:", class_num

    return node_num, class_num

def collectGraph_train(node_num, class_num, feat_dim = 256, knn = 10, suffix = '.f.npy'):
    '''
    (training dataset)
    collect info. about graph including: node, label, feature, neighborhood(adjacent) relationship.
    neighborhood(adjacent) relationship are constructed based on similarity between features.
    '''

    label = np.empty((node_num,1), dtype=np.int64)
    feature_map = np.zeros((node_num, feat_dim))
    adj_lists = defaultdict(set)

    src_folder = '/path/to/train_feature'
    class_folders = os.listdir(src_folder)
    idx = 0
    for c, folder in enumerate(class_folders):
        path = os.path.join(src_folder, folder)
        items = filter(lambda f: f.endswith(suffix), os.listdir(path))
        item_num = len(items)
        for i in range(item_num):
            item = items[i]
            ind_i = i + idx
            feature = np.load(os.path.join(path, item))
            feature_map[ind_i,:] = feature
            label[ind_i] = c
        idx += item_num

    similarity = np.dot(feature_map, feature_map.T)
    sort_id = np.argsort(-similarity, axis=1)
    for n in range(node_num):
        for k in range(1, knn+1):
            adj_lists[n].add(sort_id[n,k])

    return label, feature_map, adj_lists

def collectGraph_test(feature_path, node_num, feat_dim = 256, knn = 10, suffix = '.f.npy'):
    print "node num.:", node_num

    feature_map = np.zeros((node_num, feat_dim), dtype=np.float32)
    for n in range(node_num):
        feature_map[n,:] = np.load(os.path.join(feature_path, str(n) + suffix))
    similarity = np.dot(feature_map, feature_map.T)
    sort_id = np.argsort(-similarity, axis=1)
    adj_lists = defaultdict(set)
    for n in range(node_num):
        for k in range(1, knn+1):
            adj_lists[n].add(sort_id[n,k])
    return feature_map, adj_lists

if __name__ == "__main__":
    node_num, class_num = removeIsolated(suffix = '.frmac.npy')
