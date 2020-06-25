import numpy as np
import cv2
import matplotlib.pyplot as plt

def homography(img1,img2, plot ):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, 2)

    good = []
    pts_coords1 = []
    pts_coords2 = []
    for m, n in matches:
        if (m.distance < 0.95 * n.distance):
            if (kp1[m.queryIdx].pt[0]<108 or kp1[m.queryIdx].pt[0]>540 ):
                if (kp1[m.queryIdx].pt[1]<111 or kp1[m.queryIdx].pt[1]>840):
                    good.append([m])
                    pts_coords1.append(kp1[m.queryIdx].pt)
                    pts_coords2.append(kp2[m.trainIdx].pt)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if plot :
        plt.imshow(img3), plt.show()
    r = np.array(pts_coords1)
    s = np.array(pts_coords2)
    h, status = cv2.findHomography(r, s, cv2.RANSAC)
    return h