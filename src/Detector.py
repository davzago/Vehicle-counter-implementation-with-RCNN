import cv2 
import os

class Detector():
    def __init__(self, path_to_frame_folder):
        self.rects = []
        self.images = []
        frames = os.listdir(path_to_frame_folder)
        for i in frames:
            img = cv2.imread(path_to_frame_folder + '/' + '+i)
            self.images.append(img)
        print(frames)



    #def frame_diff(self):
        