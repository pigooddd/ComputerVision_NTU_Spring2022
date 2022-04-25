from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    
    '''
    #test code
    #print(vocab.shape)
    
    #from get_tiny_images import get_tiny_images
    #image_feats = get_tiny_images(image_paths)
    
    #img = cv2.imread(image_paths[1], 0)
    #print(dsift(img, step=[step_row, step_col], fast=True)[1].shape)
    #print("1. ",len(sift_features))
    #print("2. ",image_feats[i].sum())
    '''
    
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    print("geting bags of sifts")
    #load vocab
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    #create image_feats
    N = len(image_paths)
    d = len(vocab)
    image_feats = np.zeros((N, d)) 
    step_row = 2
    step_col = 2
    
    for i, path in enumerate(image_paths):
        #find sift_features (d = 128)
        img = cv2.imread(path, 0)
        sift_features = dsift(img, step=[step_row, step_col], fast=True)[1]
        #assign each local feature to its nearest cluster center and build a histogram
        fea_voc_distance =  distance.cdist(sift_features,vocab, 'euclidean')        
        for j in range(len(fea_voc_distance)):
            image_feats[i][fea_voc_distance[j].argmin()] += 1
                  
    #normalization
    #image_feats /= np.linalg.norm(image_feats, axis=1, keepdims=True) #L2
    image_feats /= np.linalg.norm(image_feats, ord=1, axis=1, keepdims=True) #L1
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
