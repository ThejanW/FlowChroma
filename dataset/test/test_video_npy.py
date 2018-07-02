# /home/chamath/flowchroma/lab_records/lab_record_00000.npy

import numpy as np
import cv2
from skimage import color

video = np.load('/home/chamath/flowchroma/lab_records/lab_record_02104.npy')
for i in range(video.shape[0]):
    frame = video[i]
    L = np.multiply(frame[:,:,0]+1, 50)
    A = np.multiply(frame[:,:,1], 128)
    B = np.multiply(frame[:,:,2], 128)
    new_frame = np.empty((240,320,3))
    new_frame[:,:,0] = L
    new_frame[:,:,1] = A
    new_frame[:,:,2] = B
    new_frame = color.lab2rgb(new_frame)
    cv2.imshow('image', new_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



