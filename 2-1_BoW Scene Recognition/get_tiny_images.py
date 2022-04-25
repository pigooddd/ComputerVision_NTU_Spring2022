from PIL import Image
import numpy as np
import cv2
from scipy import stats

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    
    '''
    #test code
    
    #print(len(image_paths))
    #print(image_paths[0])
    img = cv2.imread(image_paths[0], 0)#.astype(np.float32)
    #print(img.shape, type(img))
    #cv2.imshow("my image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows
    img = cv2.resize(img, (16,16), interpolation=cv2.INTER_AREA)
    #print(img.shape, type(img))
    #cv2.imshow("my image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows
    img = img.flatten()
    #print(img.shape, type(img))
    
    #print((tiny_images[3]**2).sum())
    '''
    
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #Create image features (without normalization) (Acc. = 0.19)
    N = len(image_paths)
    w = 16
    d = w**2
    tiny_images = np.zeros((N, d)) 
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (w, w), interpolation=cv2.INTER_AREA).flatten()
        tiny_images[i] = img

    # Make the image features zero mean and unit length (L2) (normalization) (Acc. = 0.23)
    tiny_images = [stats.zscore(tiny_images[i]) for i in range(N)]
           
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
