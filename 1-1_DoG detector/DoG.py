import numpy as np
import cv2



class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####

        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        
        gaussian_images.append(image)
        for i in range(4):
            gaussian_images.append(cv2.GaussianBlur (image, (0, 0), self.sigma**(i+1)))


        image_small = cv2.resize(gaussian_images[4], (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_NEAREST)
        
        #print(image.shape)
        
        gaussian_images.append(image_small)
        for i in range(4):
            gaussian_images.append(cv2.GaussianBlur (image_small, (0, 0), self.sigma**(i+1)))


        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        

        
        for i in range(4):
            dog_images.append(cv2.subtract(gaussian_images[i+1], gaussian_images[i]))
        
        for i in range(4):
            dog_images.append(cv2.subtract(gaussian_images[i+6], gaussian_images[i+5]))
            


        
        """
        #normalization
        for i in range(len(dog_images)):
            norm_denominator = np.max(dog_images[i])-np.min(dog_images[i])
            norm_min = np.min(dog_images[i])
            dog_images[i]=(dog_images[i]-norm_min)/norm_denominator*255        
        #plot
        xxx=1
        for i in dog_images:
            cv2.imwrite(str(xxx)+".png",i)
            xxx+=1
        """

        
        
        
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        
        
        keypoints = []
        
        #method 2- brutal by "and" (~54/2 comparsion/pixel)
        pyramid = [1, 2, 5, 6]
        for i in pyramid:
            for row in range(dog_images[i].shape[0]-2):
                for column in range(dog_images[i].shape[1]-2):
                    value = dog_images[i][row+1][column+1]
                    if (abs(value) > self.threshold and ((\
                    
                    #upper
                    value>=dog_images[i+1][row+1-1][column+1+1] and value>=dog_images[i+1][row+1+0][column+1+1]and value>=dog_images[i+1][row+1+1][column+1+1]\
                    and value>=dog_images[i+1][row+1-1][column+1+0]and value>=dog_images[i+1][row+1+0][column+1+0]and value>=dog_images[i+1][row+1+1][column+1+0]\
                    and value>=dog_images[i+1][row+1-1][column+1-1]and value>=dog_images[i+1][row+1+0][column+1-1]and value>=dog_images[i+1][row+1+1][column+1-1]\
                    
                    #median
                    and value>=dog_images[i+0][row+1-1][column+1+1]and value>=dog_images[i+0][row+1+0][column+1+1]and value>=dog_images[i+0][row+1+1][column+1+1]\
                    and value>=dog_images[i+0][row+1-1][column+1+0]and value>=dog_images[i+0][row+1+0][column+1+0]and value>=dog_images[i+0][row+1+1][column+1+0]\
                    and value>=dog_images[i+0][row+1-1][column+1-1]and value>=dog_images[i+0][row+1+0][column+1-1]and value>=dog_images[i+0][row+1+1][column+1-1]\
                    
                    #lower
                    and value>=dog_images[i-1][row+1-1][column+1+1]and value>=dog_images[i-1][row+1+0][column+1+1]and value>=dog_images[i-1][row+1+1][column+1+1]\
                    and value>=dog_images[i-1][row+1-1][column+1+0]and value>=dog_images[i-1][row+1+0][column+1+0]and value>=dog_images[i-1][row+1+1][column+1+0]\
                    and value>=dog_images[i-1][row+1-1][column+1-1]and value>=dog_images[i-1][row+1+0][column+1-1]and value>=dog_images[i-1][row+1+1][column+1-1]\
                    
                    ) or (\
                    
                    #upper
                    value<=dog_images[i+1][row+1-1][column+1+1]and value<=dog_images[i+1][row+1+0][column+1+1]and value<=dog_images[i+1][row+1+1][column+1+1]\
                    and value<=dog_images[i+1][row+1-1][column+1+0]and value<=dog_images[i+1][row+1+0][column+1+0]and value<=dog_images[i+1][row+1+1][column+1+0]\
                    and value<=dog_images[i+1][row+1-1][column+1-1]and value<=dog_images[i+1][row+1+0][column+1-1]and value<=dog_images[i+1][row+1+1][column+1-1]\
                    
                    #median
                    and value<=dog_images[i+0][row+1-1][column+1+1]and value<=dog_images[i+0][row+1+0][column+1+1]and value<=dog_images[i+0][row+1+1][column+1+1]\
                    and value<=dog_images[i+0][row+1-1][column+1+0]and value<=dog_images[i+0][row+1+0][column+1+0]and value<=dog_images[i+0][row+1+1][column+1+0]\
                    and value<=dog_images[i+0][row+1-1][column+1-1]and value<=dog_images[i+0][row+1+0][column+1-1]and value<=dog_images[i+0][row+1+1][column+1-1]\
                    
                    #lower
                    and value<=dog_images[i-1][row+1-1][column+1+1]and value<=dog_images[i-1][row+1+0][column+1+1]and value<=dog_images[i-1][row+1+1][column+1+1]\
                    and value<=dog_images[i-1][row+1-1][column+1+0]and value<=dog_images[i-1][row+1+0][column+1+0]and value<=dog_images[i-1][row+1+1][column+1+0]\
                    and value<=dog_images[i-1][row+1-1][column+1-1]and value<=dog_images[i-1][row+1+0][column+1-1]and value<=dog_images[i-1][row+1+1][column+1-1]\
                                        
                    ))):
                        if(i == 5 or i == 6):
                            keypoints.append([(row+1)*2, (column+1)*2])
                        else:
                            keypoints.append([row+1, column+1])
        
                    
                    
                     
        
        
        keypoints = np.array(keypoints)


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        # sort 2d-point by y, then by x
        keypoints = np.unique(keypoints, axis=0)
        #keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        

        return keypoints
        
        
        
        
        
        
        
        
        
        
