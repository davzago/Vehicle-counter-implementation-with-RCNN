import cv2


def video_to_frame(path_to_video, path_to_frame_folder):
    vidcap = cv2.VideoCapture(path_to_video)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(path_to_frame_folder + "/%d.jpg" % count, image)    
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

#video_to_frame("data/video3.MOV", "data/frames")