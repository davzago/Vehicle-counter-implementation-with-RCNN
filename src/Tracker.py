from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Tracker():
    def __init__(self, max_dis_frames=20, min_travel_distance=60):
        # initialize the id that will be given to objects (the first id is 0)
        self.objectID = 0
        # initializing the dictionary wich will contain the centers of the rectangles in the present frame
        self.objects = OrderedDict()
        # dictionary containing all the objects that disappears for a number of frames <= max_dis_frames
        self.dis_objects = OrderedDict()
        self.max_dis_frames = max_dis_frames
        self.initial_position = OrderedDict()
        self.min_travel_distance = min_travel_distance
        self.vheicle_count = 0

    # method used to register a new object 
    def register_object(self, center):
        self.objects[self.objectID] = center
        self.dis_objects[self.objectID] = 0
        self.initial_position[self.objectID] = center
        self.objectID += 1
    
    # method uset to unregister an object absent from the video for more than max_dis_frames
    def unregister(self, objectID):
        del self.objects[objectID]
        del self.dis_objects[objectID]
        if objectId in self.initial_position:
            del self.initial_position[objectID]
    
    # core method that will be called for each frame of the video and is used to
    # keep track of the objects
    def update(self, bounding_boxes):
        # first we check if there are no bounding boxes in the current image,
        # if so for each registered object we increment the number of frames in which it is absent
        # if the object is absent for more than max_dis_frames we unregister it
        if len(bounding_boxes) == 0:
            for obj in list(self.dis_objects.keys()):
                self.dis_objects[obj] += 1
                if self.dis_objects[obj] > self.max_dis_frames:
                    self.unregister(obj)
            return self.objects

        # otherwise we calculate the center of every bounding box
        centers = np.zeros((len(bounding_boxes), 2), dtype="int")

        for i, (x, y, w, h) in enumerate(bounding_boxes):
            cx = int(x+w/2)
            cy = int(y+h/2)
            centers[i] = (cx, cy)

        # if there are no objects registered we just need to register the new centers
        if len(self.objects) == 0:
            for i in range(0,len(centers)):
                self.register_object(centers[i])
        
        # otherwise we need to check if a center corresponds to an already registered one 
        else:
            # first we calculate the distance between each new center and each registered center obtaining a matrix
            objectIDs = list(self.objects.keys())
            object_centers = list(self.objects.values())
            distance = dist.cdist(np.array(object_centers), centers)
            
            # then we find the minimum value for each row ad order the rows based on its minimum value
            rows = distance.min(axis=1).argsort()
            # we find the column index of the minimum value and then adjust it based on how we ordered rows
            cols = distance.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            # for each zipped col and rows
            for row, col in zip(rows, cols):
                # we check if we already used the row or the col, if so we skip this cycle 
                if row in used_rows or col in used_cols:
                    continue
                # if row and col are not used we update the object corresponding to the row
                # with the new center corresponding to the col and set the frames disappeared to 0
                objectID = objectIDs[row]
                self.objects[objectID] = centers[col]
                self.dis_objects[objectID] = 0
                used_rows.add(row)
                used_cols.add(col)
            # we update the vheicle count after updating the distances traveled by the register objects
            for key in objectIDs:
                if key in self.initial_position:
                   d = dist.euclidean(self.initial_position[key], self.objects[key])
                   if d > self.min_travel_distance:
                       self.vheicle_count += 1 
                       del self.initial_position[key]         
            # the unused rows are disappeared object so we update dis_object with the id corresponding to the row
            # the unused cols are new objects that need to be registered       
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
    




