import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
import time


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    
#    show image    
#    plt.figure(0)
#    plt.imshow(img)
#    plt.show()
    
#   show image size    
#    h, w = img.shape
#    print(h, w)
#    print(img.shape[0])
#    print(img.shape[1])
#    print(img.shape)
    
    DoG = Difference_of_Gaussian(threshold = 5)
    start = time.time()
    keypoints = DoG.get_keypoints(image = img)
    
    #print(keypoints)
    print(keypoints.shape)
    #cv2.imwrite("graph.png", img)
    #plot_keypoints(img, keypoints, "./graph2.png")
    
    end = time.time()
    print("Time",end-start)
    
if __name__ == '__main__':
    main()
