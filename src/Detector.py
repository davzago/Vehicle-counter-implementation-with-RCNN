import cv2 
import os
import re
import numpy as np

class Detector():
    def __init__(self, path_to_frame_folder):
        self.rects = []
        self.images = []
        self.diff_images = []
        frames = os.listdir(path_to_frame_folder)
        frames.sort(key=lambda f: int(re.sub('\D', '', f)))
        for i in frames:
            print(i)
            img = cv2.imread(path_to_frame_folder + '/' + i)
            self.images.append(img)

    # loops over the frames of the video to obtain the difference between consecutive frames    
    def img_diff(self):
        for i in range (0,len(self.images)-1):
            # first we convert the frame to grayscale
            frame1 = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(self.images[i+1], cv2.COLOR_BGR2GRAY)
            # then we compute the difference between the 2 images
            diff = cv2.absdiff(frame1,frame2)
            self.diff_images.append(diff)
    
    # loops over the difference images and applies thresholding and dilate in order to find contours
    # thresholding is basically set a threshold, the pixels which value is lower than the threshold are set to 0
    # the ones which values is higher are set to 255
    # dilatation is used in order to unify fragmented regions and help the detection
    def threshold_and_dilate(self):
        for i in range(0,len(self.diff_images)):
            _, thresh = cv2.threshold(self.diff_images[i], 100, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3),np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            self.diff_images[i] = dilated

    def save_diff_images(self, path):
        is_dir = os.path.isdir(path) 
        if is_dir:
            count = 0
            for img in self.diff_images:
                cv2.imwrite(path + '/%d.jpg' % count, img)
                print("img %d done" %count)
                count += 1
        else: print("select an existing folder")


detec = Detector("data/frames")
detec.img_diff()
detec.threshold_and_dilate()
detec.save_diff_images("data/diff")
 
        