import cv2 
import os
import re
import numpy as np

class Detector():
    def __init__(self, path_to_frame_folder):
        self.rects = []
        self.diff_images = []
        frames = os.listdir(path_to_frame_folder)
        frames.sort(key=lambda f: int(re.sub('\D', '', f)))
        img1 = 0
        img2 = cv2.imread(path_to_frame_folder + '/' + str(0) + ".jpg")
        for i in range(0,len(frames)-1):
            img1 = img2
            img2 = img2 = cv2.imread(path_to_frame_folder + '/' + str(i+1) + ".jpg")
            diff = self.img_diff(img1, img2)
            self.threshold_and_dilate(diff)

    # loops over the frames of the video to obtain the difference between consecutive frames    
    def img_diff(self, img1, img2):
        # first we convert the frame to grayscale
        frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # then we compute the difference between the 2 images
        diff = cv2.absdiff(frame1,frame2)
        return diff
    
    # loops over the difference images and applies thresholding and dilate in order to find contours
    # thresholding is basically set a threshold, the pixels which value is lower than the threshold are set to 0
    # the ones which values is higher are set to 255
    # dilatation is used in order to unify fragmented regions and help the detection
    def threshold_and_dilate(self, diff):
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        self.diff_images.append(dilated)

    def save_diff_images(self, path):
        is_dir = os.path.isdir(path) 
        if is_dir:
            count = 0
            for img in self.diff_images:
                cv2.imwrite(path + '/%d.jpg' % count, img)
                print("img %d done" %count)
                count += 1
        else: print("select an existing folder")

    def draw_bbox(self):
        for i in range(0, len(self.diff_images)):
            contours, _ = cv2.findContours(self.diff_images[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(self.diff_images[i], contours, -1, (127,200,0), 2)
            rec = []
            for _, poly in enumerate(contours):
                if cv2.contourArea(poly) > 600: # this makes so we only take useful contours, avoids bbox with a small area
                    x,y,w,h = cv2.boundingRect(poly)
                    #top_left = (x,y)
                    #bottom_right = (x+w,y+h)
                    rec.append([x, y, w, h])
                    #cv2.rectangle(self.diff_images[i], top_left, bottom_right, (127,200,0), 1)                 
                        
            
            self.rects.append(rec)
    
    def detect(self, path_to_diff):
        self.draw_bbox()
        self.save_diff_images(path_to_diff)





#detec = Detector("data/frames")
#detec.detect("data/diff")
#for i,((x1,y1),(x2,y2)) in enumerate(detec.rects[875]):
#    print(i, x1, y1, x2, y2)
 
        
        