import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


def extrinsic_extraction(H,K):
    k_inv = np.linalg.inv(K)
    mRt = np.dot(k_inv,H)
    U, D, V = np.linalg.svd(mRt)
    mRT =np.dot(U,V)
    r1 = mRt[:,0]
    r2 = mRt[:,1]
    t  = mRt[:,2]
    print(np.inner(r1,r2))
    print(np.linalg.norm(r1))
    r3 = np.cross(r1,r2)
    R = np.column_stack((r1,r2,r3))
    return (R,t)
