# world library
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from progress.bar import IncrementalBar
#personal library
from Calibration import zhang_calibration
from drawing import drawing

if __name__== "__main__":
#----------------------- CALIBRATION USING A ZHANG APPROACH -----------------------#
    CHECKERBOARD = (7, 7)
    images = glob.glob("./Data/Calibration/*.jpg")
    K = zhang_calibration(images, CHECKERBOARD)  #calibration by Zhang
#------------------------ END OF CALIBRATION---------------------------------------#
    print ("=========== STANDARD VIEW AND VIDEO LOAD =========== \n")
    standard_view = cv2.imread("./Data/standard_view.jpg")
    #plt.imshow(standard_view), plt.show()
    video = cv2.VideoCapture ("./Data/video.mp4")
    frames_amount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    progress_bar =  IncrementalBar(' ', max =int(frames_amount))
    print ("=========== START VIDEO PROCESSING =========== \n")
    img_array=[]
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == False:
            break
        frame = drawing(standard_view,frame,K)
        img_array.append(frame)
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame.shape[1], frame.shape[0]))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        progress_bar.next()
    progress_bar.finish()



