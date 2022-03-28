import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        (X)method1 brutal (~4 s)
        (O)method2 look up table (~3 s)
        
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        
        """
        #1 create spacial kernel
        """

        spacial_kernel = np.zeros((self.wndw_size,self.wndw_size))
        
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                spacial_kernel[i][j] = np.exp((-(i-self.pad_w)**2-(j-self.pad_w)**2)/(2*(self.sigma_s)**2))

        """
        #2 create LUT for range kernel
        """
        

        two_r_square = 2*self.sigma_r**2
        range_kernel_LUT = np.arange(0,510,dtype=np.float64)
        range_kernel_LUT[:256] = np.exp(-(range_kernel_LUT[:256]/255)**2/two_r_square)
        range_kernel_LUT[256:] = range_kernel_LUT[254:0:-1]

        """
        #3 start filtering
        """
        
        output = np.zeros(img.shape)
        range_kernel = np.zeros((self.wndw_size,self.wndw_size))


        #run over all the pixel in img
        for row in range(self.pad_w, img.shape[0]+self.pad_w):
            for column in range(self.pad_w, img.shape[1]+self.pad_w):
                
                #calculate range kernel
                center_pixel = padded_guidance[row][column]
                range_kernel = padded_guidance[row-self.pad_w:row+self.pad_w+1, column-self.pad_w:column+self.pad_w+1]
                range_kernel = range_kernel_LUT[range_kernel-center_pixel]
                if (len(padded_guidance.shape)==3):
                    range_kernel = range_kernel.prod(axis=2)

                #normalize
                total_kernel = range_kernel*spacial_kernel
                total_kernel = total_kernel/total_kernel.sum()

                #calculate output
                for d in range(padded_img.shape[2]):
                    output[row-self.pad_w][column-self.pad_w][d] = (padded_img[row-self.pad_w:row+self.pad_w+1, column-self.pad_w:column+self.pad_w+1,d]*total_kernel).sum()

        return np.clip(output, 0, 255).astype(np.uint8)
        
        
