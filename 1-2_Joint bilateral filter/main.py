import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
from matplotlib import pyplot as plt
import time

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.int32)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.int32)

    f = open(args.setting_path, "r")
    variables = f.readlines()
    
    gray_conversion = np.zeros((5,3))
    
    #needed variable in .txt
    for i in [1,2,3,4,5]:
        gray_conversion[i-1]=variables[i].split(",")    
    sigma_s = int(variables[6].split(",")[1])
    sigma_r = np.float64(variables[6].split(",")[3])
        
    ### TODO ###
       
    #plt.figure(0)
    #plt.imshow(img_rgb )
    #plt.show()
    #print(img_rgb.shape)
    
    
    start = time.time()

    
    
    weight = [0.8, 0.2, 0.0]
    img_gray_conversion = img_rgb[:,:,0]*weight[0]+img_rgb[:,:,1]*weight[1]+img_rgb[:,:,2]*weight[2]
    
    #cv2.imshow("my image",img_gray_conversion)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows

    #print(img_gray_conversion.shape)
    
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray_conversion)
    cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    print(cost)
    
    
    cv2.imwrite("gray.png", img_gray_conversion)
    cv2.imwrite("RGB.png", cv2.cvtColor(jbf_out,cv2.COLOR_BGR2RGB))
    
    
    
    
    
    
    """
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
    cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    print("img_gray: ", cost)
    

    for weight in gray_conversion:
        
        img_gray_conversion = img_rgb[:,:,0]*weight[0]+img_rgb[:,:,1]*weight[1]+img_rgb[:,:,2]*weight[2]

        bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray_conversion.astype(np.int32))
        cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        print(weight, ": ", cost)
        
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #plt.figure(0)
    #plt.imshow(jbf_out)
    #plt.show()
    #print(jbf_out.shape)
    
    
    
    
    end = time.time()
    print("Time: ",end-start)


if __name__ == '__main__':
    main()
