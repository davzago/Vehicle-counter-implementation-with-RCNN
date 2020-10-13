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

        if len(self.objects) == 0:
            for i in range(0,len(centers)):
                self.register_object(centers[i])

        else:
            objectIDs = list(self.objects.keys())
            object_centers = list(self.objects.values())
            distance = dist.cdist(np.array(object_centers), centers)
            rows = distance.min(axis=1).argsort()
            cols = distance.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = centers[col]
                self.dis_objects[objectID] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(0,distance.shape[0])).difference(used_rows)
            unused_cols = set(range(0,distance.shape[1])).difference(used_cols)
            if distance.shape[0] >= distance.shape[1]:
                for row in unused_rows:
                    objectID = objectIDs[row]
                    self.dis_objects[objectID] += 1
                    if self.dis_objects[objectID] > self.max_dis_frames:
                        self.unregister(objectID)
            else:
                for col in unused_cols:
                    self.register_object(centers[col])
        return self.objects
    



