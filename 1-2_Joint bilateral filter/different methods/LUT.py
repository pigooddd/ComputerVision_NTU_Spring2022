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
        
        method1 brutal (~3.75s)
        method2 look up table (?s)
        
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        
        #print(type(img[0][0][0]))
        #print(img.shape)
        #print(padded_img.shape)
        
        """
        #1 create spacial kernel
        """
        
        #self.wndw_size = 3
        #self.pad_w = 1
        #self.sigma_s = 1 
        
        
        spacial_kernel = np.zeros((self.wndw_size,self.wndw_size))
        
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                spacial_kernel[i][j] = np.exp((-(i-self.pad_w)**2-(j-self.pad_w)**2)/(2*(self.sigma_s)**2))
                

        
        #print(self.wndw_size, self.sigma_s)
        #print(spacial_kernel)
        
        """
        #2 create range kernel
        """
        range_kernel = np.zeros((self.wndw_size,self.wndw_size))
        
        #create look up table for exp. cal.
        two_r_square = 2*self.sigma_r**2
        range_kernel_LUT = np.arange(0,510,dtype=np.float64)
        range_kernel_LUT[:256] = np.exp(-(range_kernel_LUT[:256]/255)**2/two_r_square)
        range_kernel_LUT[256:] = range_kernel_LUT[254:0:-1]

        #print(range_kernel_LUT)
        #print(type(range_kernel_LUT[0]))
        #print(type(padded_guidance[0][0][0]))

        
        """
        #3 start filtering
        """
        
        output = np.zeros(img.shape)
        #normalize pixel values
        #normalized_padded_guidance = padded_guidance/255
        
        
        #print(padded_guidance.shape)
        #print(self.pad_w, img.shape[0]+self.pad_w)
        #print(normalized_padded_guidance[6-self.pad_w:6+self.pad_w+1, 6-self.pad_w:6+self.pad_w+1].shape)
        #print(len(normalized_padded_guidance.shape))
        #print(output.shape)

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
        
     
        
        #print(output.shape)
        return np.clip(output, 0, 255).astype(np.uint8)
        
        
        
        
        
#python3 eval.py --image_path './testdata/ex.png' --gt_bf_path './testdata/ex_gt_bf.png' --gt_jbf_path './testdata/ex_gt_jbf.png'


