import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(99)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        print("img: ", idx+2)
        im1 = imgs[idx] #已經在canvas上
        im2 = imgs[idx + 1] #未在canvas上

        # TODO: 1.feature detection & matching (ref: https://zhuanlan.zhihu.com/p/143446523, https://www.jianshu.com/p/ed57ee1056ab)
        #初始化ORB detector
        orb = cv2.ORB_create()
        # 用ORB找keypoints和descriptors
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        #初始化brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #找im1和im2的match點
        matches = bf.match(des1, des2)
        #按照match程度(distance)排列
        matches = sorted(matches, key = lambda x:x.distance)
        #取match程度排名前50%
        good = matches[:int(len(matches) * 0.3)] #0.3
        #取座標(list包tuple座標)      
        kp1_coordinate = np.array([kp1[mat.queryIdx].pt for mat in good])
        kp2_coordinate = np.array([kp2[mat.trainIdx].pt for mat in good])
        kp_coordinate_size = len(kp1_coordinate)
        
        # TODO: 2. apply RANSAC to choose best H
        iteration_num = 20 #20
        selected_points = kp_coordinate_size*3//10 #kp_coordinate_size*3//10 12
        distance_threshold = 2.67 #2.67
        inlier_size_threshold = kp_coordinate_size*0.95 #0.95
        max_inlier_size = 0
        max_inlier = np.zeros(kp_coordinate_size).astype(bool)
        
        for i in range(iteration_num):
            # (a) Select 4 correspondences and compute H
            random_point = np.random.choice(np.arange(kp_coordinate_size), selected_points, replace=False)
            H =  solve_homography(kp2_coordinate[random_point], kp1_coordinate[random_point])#im1是canvas
            # (b) Calculate the distance d⊥ for each putative match
            #計算kp1預測值
            transformed_kp1_coordinate = np.dot(H, np.vstack((kp2_coordinate.T, np.ones(kp_coordinate_size))))#np.float64
            transformed_kp1_coordinate = np.round(transformed_kp1_coordinate[0:2,:]/transformed_kp1_coordinate[2]).T
            #計算kp1預測值與kp1距離           
            distance = np.sqrt(np.sum(np.square(kp1_coordinate-transformed_kp1_coordinate),axis=1))#euclidean

            # (c) Compute the number of inliers consistent with H (d⊥<t)
            inliers = distance <= distance_threshold
            inlier_size = sum(inliers)
            #紀錄最大inliers數量和其inliers
            if (inlier_size >= max_inlier_size):
                max_inlier_size = inlier_size
                max_inlier = inliers
                print("it: ", i," inlier size racio", max_inlier_size/kp_coordinate_size)
                # print(max_inlier_size)
                if (inlier_size >= inlier_size_threshold):
                    print("arrive threshold")
                    break

        # TODO: 3. chain the homographies
        H =  solve_homography(kp2_coordinate[max_inlier], kp1_coordinate[max_inlier])
        #kp1和kp2的座標剛好是x-y表示(跟row-col顛倒)，所以跟part1-3一樣是用x-y算H，最後要warping圖時才換回row-col
        #因此不須使用, H[:,[0,1]] = H[:,[1,0]]
        last_best_H = np.dot(last_best_H, H)
        
        # TODO: 4. apply warping        
        warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
        
        out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    
    #test1, sample picture
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
    
    # #test2, indoor view
    # FRAME_NUM = 4
    # imgs = [cv2.imread('../resource/indoor{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    # output4_indoor = panorama(imgs)
    # cv2.imwrite('output4_indoor.png', output4_indoor)

    #test3, 270 view
    # FRAME_NUM = 11
    # imgs = [cv2.imread('../resource/270{:d}.jpg'.format(x)) for x in range(5, FRAME_NUM + 1-1)]#(5, FRAME_NUM + 1-2)
    # output4_270 = panorama(imgs)
    # cv2.imwrite('output4_270.png', output4_270)
    
    #test4, shift view
    # FRAME_NUM = 3
    # imgs = [cv2.imread('../resource/shift{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]#(5, FRAME_NUM + 1-2)
    # output4_shift = panorama(imgs)
    # cv2.imwrite('output4_shift.png', output4_shift)