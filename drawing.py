from homography import homography
import numpy as np
import first_way as fw
import cv2
def drawing(img1,img2, K,plot=False):
    H = homography(img1,img2,plot)
    R ,t = fw.extrinsic_extraction(H,K)
    Rt = np.column_stack((R,t))
    h_complete = np.dot(K,Rt)
    lists = np.argwhere(img1<100)
    for el in lists:
          X = np.array((el[1], el[0], -9000, 1))
          X0 = np.array((el[1], el[0], 0, 1))
          x = np.dot(h_complete, X)
          x = (x / x[2])
          x0 = np.dot(h_complete, X0)
          x0 = (x0 / x0[2])
          if (x[0] > 0 and x[1] > 0):
              if (x[0] < img2.shape[1] and x[1] < img2.shape[0]):
                  cv2.line(img2, (int(x0[0]), int(x0[1])), (int(x[0]), int(x[1])), (0, 0, 0))
    return img2