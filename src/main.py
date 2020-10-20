from Detector import Detector
from Tracker import Tracker
import cv2
import time

detector = Detector("data/frames")
tracker = Tracker()
detector.detect("data/diff")
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

s_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

s_search_path = "data/SSearch"
times = []
for i in range(0,len(detector.diff_images)+1):
    img = cv2.imread("data/frames/%d.jpg" %i)
    s_search.setBaseImage(img)
    s_search.switchToSelectiveSearchFast()
    start = time.time()
    rectangles = s_search.process()
    end = time.time()
    times.append(end-start)
    for (x, y, h, w) in rectangles:
        cv2.rectangle(img, (x, w), (x+w, y+h), (0, 255, 0), 1)
    cv2.imwrite(s_search_path +'/%d.jpg' % i, img)

print("The mean computation time of selective search is:",sum(times)/len(times))
    

    

