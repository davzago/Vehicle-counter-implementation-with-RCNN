# Veichle counting and classification

## Methods 

- **Frame Differencing:** consists in using 2 neighbor frames 2 determine the moving objects, we use the difference of the 2 frames to obtain contours for the objects.
  This method is useful because its unsupervised unlike deep learning methods which needs som ground truth both when building a model from scratch or when fine applying fine tuning on an already existing one (https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/) ( we could use this to obtain some bounding box to then classify the veichle)
- **Region based CNN:** this method is the base of object detection, it is based on region of interest methods which first select some areas of the image and then apply conv NN to select the objects in them, also using this method we can classify and get a better(more tight) bounding box over the object using regression (https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/?utm_source=blog&utm_medium=vehicle-detection-opencv-python, IMPLEMENTATION: https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/)
- YOLO and implementation: (https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/)
- **SlimYOLO:** basically taking a Yolo NN, training it and prune the less important connection based on the L1 norm on the weights, after this we fine tune the net and evaluate performance, we can decide to repeat this steps to obtain a slimmer network, this net usually performs worse than RCNN but we gain evaluation speed, this is the tradeoff (https://www.analyticsvidhya.com/blog/2019/08/introduction-slimyolov3-real-time-object-detection/?utm_source=blog&utm_medium=vehicle-detection-opencv-python)

- 