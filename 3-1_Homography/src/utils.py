import numpy as np
import numpy.ma as ma

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2*N,9))
    for i in range(N):
        #first row
        A[2*i][3] = -u[i][0]
        A[2*i][4] = -u[i][1]
        A[2*i][5] = -1
        
        A[2*i][6] = u[i][0]*v[i][1]
        A[2*i][7] = u[i][1]*v[i][1]
        A[2*i][8] = v[i][1]
        #second row
        A[2*i+1][0] = u[i][0]
        A[2*i+1][1] = u[i][1]
        A[2*i+1][2] = 1
        
        A[2*i+1][6] = -u[i][0]*v[i][0]
        A[2*i+1][7] = -u[i][1]*v[i][0]
        A[2*i+1][8] = -v[i][0]
    # TODO: 2.solve H with A
    u, s, vh = np.linalg.svd(A)
    #extract H
    H = vh[-1].reshape(3,3)
    #normalize H with |h|=1
    H = H/np.linalg.norm(H)
    
    return H

#warping的ymax和xmax是不包含的，只會到max-1
def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    X,Y = np.mgrid[xmin:xmax:1, ymin:ymax:1]
    W = np.ones((xmax-xmin)*(ymax-ymin)).astype(np.int32)
    homocoordinate_pairs = np.vstack((X.flatten(), Y.flatten(), W))
    # print(homocoordinate_pairs[-1])

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        #計算transform對應的位置
        transformed_homocoordinate_pairs = np.dot(H_inv, homocoordinate_pairs)
        #計算x-y實際值(除w項)
        xy_pairs = homocoordinate_pairs[0:2,:]
        xy_transformed_pairs = np.round(transformed_homocoordinate_pairs[0:2,:]/transformed_homocoordinate_pairs[2]).astype(np.int32)
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        #在範圍外(要刪掉的)顯示True
        xy_pairs_mask = (xy_transformed_pairs[0]<0) + (xy_transformed_pairs[0]>(w_src-1)) \
            + (xy_transformed_pairs[1]<0) + (xy_transformed_pairs[1]>(h_src-1))
            
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        xy_pairs = xy_pairs[:,~xy_pairs_mask]
        xy_transformed_pairs = xy_transformed_pairs[:,~xy_pairs_mask]

        # TODO: 6. assign to destination image with proper masking
        #於dst覆蓋src圖片(img上的x-y會顛倒(row-col))
        dst[xy_pairs[1], xy_pairs[0]] =\
            src[xy_transformed_pairs[1], xy_transformed_pairs[0]]

        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        #計算transform對應的位置
        transformed_homocoordinate_pairs = np.dot(H, homocoordinate_pairs)
        #計算x-y實際值(除w項)
        xy_pairs = homocoordinate_pairs[0:2,:]
        xy_transformed_pairs = np.round(transformed_homocoordinate_pairs[0:2,:]/transformed_homocoordinate_pairs[2]).astype(np.int32)
        # print(xy_homocoordinate_pairs.shape, xy_transformed_homocoordinate_pairs.shape)
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        # x_pairs_mask = ma.masked_outside(xy_transformed_pairs[0], 0, w_dst-1).mask#x的boundary
        # y_pairs_mask = ma.masked_outside(xy_transformed_pairs[1], 0, h_dst-1).mask#y的boundary
        # xy_pairs_mask = ma.masked_where(xy_transformed_pairs[0]<0 or  xy_transformed_pairs[0]>w_dst-1\
        #     or xy_transformed_pairs[1]<0 or xy_transformed_pairs[1]>h_dst-1, xy_transformed_pairs)
        
        #在範圍外(要刪掉的)顯示True
        xy_pairs_mask = (xy_transformed_pairs[0]<0) + (xy_transformed_pairs[0]>(w_dst-1)) \
            + (xy_transformed_pairs[1]<0) + (xy_transformed_pairs[1]>(h_dst-1))
        
        # xy_pairs = ma.array(xy_pairs, mask = np.tile(xy_pairs_mask, (2,1)))
        # xy_transformed_pairs = ma.array(xy_transformed_pairs, mask = np.tile(xy_pairs_mask, (2,1)))
        # print(xy_pairs_mask.astype(np.int32).sum(), len(transformed_homocoordinate_pairs[0]))
        # xy_transformed_pairs.view(ma.MaskedArray)
        # print(xy_pairs.mask)
        
        # TODO: 5.filter the valid coordinates using previous obtained mask
        #把範圍外(顯示True的)刪掉
        xy_pairs = xy_pairs[:,~xy_pairs_mask]
        xy_transformed_pairs = xy_transformed_pairs[:,~xy_pairs_mask]
        
        # TODO: 6. assign to destination image using advanced array indicing
        #於dst覆蓋src圖片(img上的x-y會顛倒(row-col))
        dst[xy_transformed_pairs[1], xy_transformed_pairs[0]] =\
            src[xy_pairs[1], xy_pairs[0]]
        
        
        pass

    return dst
