from Detector import Detector
from Tracker import Tracker
import cv2
import time
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

def nms(rectangles, overlap_threshold):
    x1 = rectangles[:,0]
    y1 = rectangles[:,1]
    x2 = rectangles[:,2]
    y2 = rectangles[:,3]
    labels = rectangles[:,4]  
    prob = rectangles[:,5]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(prob)

    pick = []

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indexes[:last]]

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    for k in pick:
        rectangles[k,2] = rectangles[k,2] - rectangles[k,0]
        rectangles[k,3] = rectangles[k,3] - rectangles[k,1]    
    return rectangles[pick].astype("int")


#detector = Detector("data/frames")
tracker = Tracker()
#detector.detect("data/diff")
path = "data/results"

"""for i in range(0,len(detector.rects)):
    centers = tracker.update(detector.rects[i])
    img = cv2.imread("data/frames/%d.jpg" %i)
    for j in range(0,len(detector.rects[i])):
        cv2.rectangle(img, detector.rects[i][j][0], detector.rects[i][j][1], (0, 255, 0), 1)
    for c_id, c in centers.items():
        text = "ID{}".format(c_id)
        cv2.putText(img, text, (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (c[0],c[1]), 4, (0, 255, 0), -1)
        
    cv2.imwrite(path + '/%d.jpg' % i, img)"""

model = keras.models.load_model('models/model1k.h5')

s_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

s_search_path = "data/SSearch"
times = []
crops = []
vheicles = []
for i in range(419,423):
    img = cv2.imread("data/frames/%d.jpg" %i)
    max_height, max_width, _ = img.shape
    s_search.setBaseImage(img)
    s_search.switchToSelectiveSearchFast()
    print("starting selective search on image %d" %i)
    start = time.time()
    rectangles = s_search.process()
    end = time.time()
    times.append(end-start)
    crop = []
    count = 0
    v = []
    for (x, y , w, h) in rectangles:
        center = (int(x+w/2), int(y+h/2))
        c = cv2.getRectSubPix(img, (w, h), center)
        #crop.append(c)
        cv2.imwrite("data/crops/%d.jpg" %count, c)
        count += 1
        crop2 = []
    for j in range(0, count):
        temp_img = load_img("data/crops/%d.jpg" %j, target_size=(224, 224))
        os.remove("data/crops/%d.jpg" %j)
        temp_img = img_to_array(temp_img) 
        crop2.append(preprocess_input(temp_img))
    data = np.array(crop2)
    del crop2
    labels = model.predict(data)
    K.clear_session()
    del data
    del crop
    #l = []
    for k in range(0,len(labels)):
        if labels[k].argmax()==3:
            #cv2.imwrite("data/crops" + '/%d.jpg' % k, crop[k])
            x, y, w, h = rectangles[k]
            idx = labels[k].argmax()
            rect = (x, y, x+w, y+h, idx, labels[k, idx])
            v.append(rect)
            #l.append(labels[k])
    rect_and_labels = np.array(v)
    nms_boxes = nms(rect_and_labels, 0.3)        
    vheicles.append(nms_boxes[:,:4])
    #rec.append(rectangles)
    #for (x, y, h, w) in rectangles:
    #    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #cv2.imwrite(s_search_path +'/%d.jpg' % i, img)

print("The mean computation time of selective search is:",sum(times)/len(times))

for i in range(0,len(vheicles)):
    centers = tracker.update(vheicles[i])
    img = cv2.imread("data/frames/%d.jpg" %(419+i))
    for j in range(0,len(vheicles[i])):
        cv2.rectangle(img, (vheicles[i][j][0], vheicles[i][j][1]), (vheicles[i][j][0]+vheicles[i][j][2],vheicles[i][j][1]+vheicles[i][j][3]), (0, 255, 0), 1)
    for c_id, c in centers.items():
        text = "ID{}".format(c_id)
        cv2.putText(img, text, (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (c[0],c[1]), 4, (0, 255, 0), -1)
        
    cv2.imwrite(path + '/%d.jpg' % i, img)
    

    

