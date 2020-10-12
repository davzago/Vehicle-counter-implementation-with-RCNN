from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Tracker():
    def __init__(self, max_dis_frames=20):
        self.objectID = 0
        self.objects = OrderedDict()
        self.dis_objects = OrderedDict()
        self.max_dis_frames = max_dis_frames

    def register_object(self, center):
        self.objects[self.objectID] = center
        self.dis_objects[self.objectID] = 0
        self.objectID += 1
    
    def unregister(self, objectID):
        del self.objects[objectID]
        del self.dis_objects[objectID]
    
    def update(self, bounding_boxes):
        if len(bounding_boxes) == 0:
            for obj in range(0,self.objectID):
                self.dis_objects[obj] += 1
                if self.dis_objects[obj] > self.max_dis_frames:
                    self.unregister(obj)
            return self.objects

        centers = np.zeros((len(bounding_boxes), 2), dtype="int")

        for i,((x1,y1),(x2,y2)) in enumerate(bounding_boxes):
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            centers[i] = (cx, cy)

        



