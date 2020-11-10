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
        temp_img = img_to_array(temp_img) 
        crop2.append(preprocess_input(temp_img))
    data = np.array(crop2)
    del crop2
    labels = model.predict(data)
    K.clear_session()
    del data
    del crop
    for k in range(0,len(labels)):
        if labels[k].argmax()==3:
            #cv2.imwrite("data/crops" + '/%d.jpg' % k, crop[k])
            v.append(rectangles[k])
    vheicles.append(v)
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
    

    

