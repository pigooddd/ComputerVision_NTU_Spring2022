import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    
    #create census cost map
    Il_cost = np.zeros((h, w, 9, ch), dtype=np.bool)
    Ir_cost = np.zeros((h, w, 9, ch), dtype=np.bool)
    Il_padding = cv2.copyMakeBorder(Il, 1,1,1,1, cv2.BORDER_REPLICATE)
    Ir_padding = cv2.copyMakeBorder(Ir, 1,1,1,1, cv2.BORDER_REPLICATE)
    
    for y in range(1,h+1):
        for x in range(1,w+1):
            for i in range(0,9):
                Il_cost[y-1,x-1,i] =  Il_padding[y-1+(i//3),x-1+(i%3)] <= Il_padding[y,x]
                Ir_cost[y-1,x-1,i] =  Ir_padding[y-1+(i//3),x-1+(i%3)] <= Ir_padding[y,x]
                
    # print(Il[0:5,0:5,0])
    # print(Il_cost[3,0][:,0])

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    
    #create cost-volume
    Il_cost_volume = np.zeros((max_disp+1, h, w), dtype=np.float32)
    Ir_cost_volume = np.zeros((max_disp+1, h, w), dtype=np.float32)
    
    for d in range(0,max_disp+1): #d = 移了幾格
        cost_map = (Il_cost[:,0+d:w,:,:]^Ir_cost[:,0:w-d,:,:]).sum(axis=3).sum(axis=2)
        #np.repeat(cost_map[:,0], d).reshape(h,d)
        #Il_cost_volume[d-1,:,0:d]
        Il_cost_volume[d] = np.concatenate((np.repeat(cost_map[:,0], d).reshape(h,d),\
            cost_map),\
            axis=1) #Il往左移，看與Ir的cost => 左邊須補d個cost值
        Il_cost_volume[d] = xip.jointBilateralFilter(Il, Il_cost_volume[d], d=70, sigmaColor=4, sigmaSpace=24)
        
        #np.repeat(cost_map[:,-1], d).reshape(h,d)
        #Ir_cost_volume[d-1,:,w-d:w]
        Ir_cost_volume[d] = np.concatenate((cost_map,\
            np.repeat(cost_map[:,-1], d).reshape(h,d)),\
            axis=1) #Ir往右移，看與Il的cost => 右邊須補d個cost值
        Ir_cost_volume[d] = xip.jointBilateralFilter(Ir, Ir_cost_volume[d], d=70, sigmaColor=4, sigmaSpace=24)
        
        # print((Il_cost[:,0+d:w,:,:]^Ir_cost[:,0:w-d,:,:]).sum(axis=3).sum(axis=2).shape)
        # print((Il_cost[:,0+d:w,:,:]^Ir_cost[:,0:w-d,:,:]).sum(axis=3).sum(axis=2)[0,0])
    # print(Ir_cost_volume[5,99,-1:-50:-1])
        
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    
    Il_disp_map = Il_cost_volume.argmin(axis=0)
    Ir_disp_map = Ir_cost_volume.argmin(axis=0)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    #Left-right consistency check
    #為什麼難配對時會有hole(不是同一個東西算出來的嗎) => 因為位移相反，所以cost min不會一樣，consistency check差異出現
    for y in range(0,h):
        for x in range(0,w):
            if (Il_disp_map[y,x] != Ir_disp_map[y,x-Il_disp_map[y,x]]):
                Il_disp_map[y,x] = -1
    #Hole filling        
    Il_disp_map_padding = cv2.copyMakeBorder(Il_disp_map, 1,1,1,1, cv2.BORDER_CONSTANT, value=max_disp)
    for y in range(1,h+1):
        for x in range(1,w+1):
            if (Il_disp_map_padding[y,x] == -1):
                #to left
                Fl_x = x
                Fl = -1
                while(Fl == -1):
                    Fl_x -= 1
                    Fl = Il_disp_map_padding[y,Fl_x]
                #to right
                Fr_x = x
                Fr = -1
                while(Fr == -1):
                    Fr_x += 1
                    Fr = Il_disp_map_padding[y,Fr_x]
                    
                # Il_disp_map_padding[y,x] = min(Fl,Fr)
                Il_disp_map[y-1,x-1] = min(Fl,Fr)
            
    # print(Il_disp_map[50])
    # print(Il_disp_map_padding[51])
    
    #Weighted median filtering
    # Il_disp_map = xip.weightedMedianFilter(Il.astype(np.uint8), Il_disp_map.astype(np.uint8), r=18, sigma=1)
    #, weightType = xip.WMF_EXP
    Il_disp_map = xip.weightedMedianFilter(Il.astype(np.uint8), Il_disp_map.astype(np.uint8), r=8, sigma=38, weightType = xip.WMF_EXP)
    
    # print(Il_disp_map.max())
    # print(Il_disp_map.min()) 
           
    labels = Il_disp_map
    return labels.astype(np.uint8)
