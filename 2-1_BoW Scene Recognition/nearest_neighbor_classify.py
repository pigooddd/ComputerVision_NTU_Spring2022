from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode
from collections import Counter

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    
    '''
    #test code
    #test_predicts = train_labels[:]
       
    #test_predicts = [train_labels[test_train_distance[i].argmin()] for i in range(len(test_train_distance))]
    #print(train_labels[np.argpartition(test_train_distance[0], k-1)[:k]])
    #print(train_labels[[1,2]])
    #print(Counter(train_labels[np.argpartition(test_train_distance[5], k-1)[:k]]).most_common(1)[0][0])
    
    #for tiny_image (k = 1)
    'euclidean' => (Acc. = 0.23)
    'minkowski', p=2. => (Acc. = 0.23)
    'cityblock' => (Acc. = 0.23) *best
    'seuclidean', V=None  => (Acc. = 0.22)
    'correlation' => (Acc. = 0.23)
    'jaccard' => (Acc. = 0.07)
    'jensenshannon' => (Acc. = 0.07)
    'chebyshev' => (Acc. = 0.16)
    'canberra' => (Acc. = 0.21)
    'braycurtis'  => (Acc. = 0.22)
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #for bags_of_sifts (k = 5, 400, 100, 20)
    'euclidean' => (Acc. = 0.42) *best
    'minkowski', p=2. => (Acc. = 0.42)
    'cityblock' => (Acc. = 0.27) 
    '''
    
    
    
    #建立test_train距離的表格
    test_train_distance =  distance.cdist(test_image_feats,train_image_feats, 'cityblock')
    #k個人投票
    k = 5 #BoS要5，GTI要1
    train_labels = np.array(train_labels)
    #test_predicts = [train_labels[np.argpartition(test_train_distance[i], k-1)[:k]][0] for i in range(len(test_train_distance))]
    test_predicts = [Counter(train_labels[np.argpartition(test_train_distance[i], k-1)[:k]]).most_common(1)[0][0] for i in range(len(test_train_distance))]
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
