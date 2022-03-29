import numpy as np
import cv2

img = cv2.imread('./8.png', 0).astype(np.int32)


print(img.shape)
