from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time
import cv2

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    ##################################################################################
    # TODO:                                                                          #
    # Load images from the training set. To save computation time, you don't         #
    # necessarily need to sample from all images, although it would be better        #
    # to do so. You can randomly sample the descriptors from each image to save      #
    # memory and speed up the clustering. Or you can simply call vl_dsift with       #
    # a large step size here.                                                        #
    #                                                                                #
    # For each loaded image, get some SIFT features. You don't have to get as        #
    # many SIFT features as you will in get_bags_of_sift.py, because you're only     #
    # trying to get a representative sample here.                                    #
    #                                                                                #
    # Once you have tens of thousands of SIFT features from many training            #
    # images, cluster them with kmeans. The resulting centroids are now your         #
    # visual word vocabulary.                                                        #
    ##################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # This function will sample SIFT descriptors from the training images,           #
    # cluster them with kmeans, and then return the cluster centers.                 #
    #                                                                                #
    # Function : dsift()                                                             #
    # SIFT_features is a N x 128 matrix of SIFT features                             #
    # There are step, bin size, and smoothing parameters you can                     #
    # manipulate for dsift(). We recommend debugging with the 'fast'                 #
    # parameter. This approximate version of SIFT is about 20 times faster to        #
    # compute. Also, be sure not to use the default value of step size. It will      #
    # be very slow and you'll see relatively little performance gain from            #
    # extremely dense sampling. You are welcome to use your own SIFT feature.        #
    #                                                                                #
    # Function : kmeans(X, K)                                                        #
    # X is a M x d matrix of sampled SIFT features, where M is the number of         #
    # features sampled. M should be pretty large!                                    #
    # K is the number of clusters desired (vocab_size)                               #
    # centers is a d x K matrix of cluster centroids.                                #
    #                                                                                #
    # NOTE:                                                                          #
    #   e.g. 1. dsift(img, step=[?,?], fast=True)                                    #
    #        2. kmeans( ? , vocab_size)                                              #  
    #                                                                                #
    # ################################################################################
    
    '''
    #test code
    #vl_dsift
    #print(np.array(image_paths).shape)

    #img = cv2.imread(image_paths[0], 0)
    #print(dsift(img, step=[100,100], fast=True)[0].shape)
    
    #print(np.array((sift_features)).shape)
    #print(dsift(img, step=[500,500], fast=True)[1].shape)
    
    #print(type(sift_features[0][2]))
    print(type(vocab))
    print(type(vocab[1]))
    print(type(vocab[1][1]))
    '''
    
    '''
    Input : 
        image_paths : a list of training image path
        vocal size : number of clusters desired
    Output :
        Clusters centers of Kmeans
    '''
    print("building vocabulary")
    step_row = 50
    step_col = 50
    sift_features = []
    
    #find all sift_features (M, d = 128) 
    for path in image_paths:
        img = cv2.imread(path, 0)
        sift_features += list(dsift(img, step=[step_row, step_col], fast=True)[1])

    #find visual word vocabulary (K = vocab_size, d = 128) by kmeans
    vocab = kmeans(np.array(sift_features).astype(np.float64), vocab_size)
    
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    return vocab

