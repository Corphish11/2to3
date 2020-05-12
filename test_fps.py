import cv2
if __name__ == '__main__' :

    video = cv2.VideoCapture("output1_1.mp4");
    
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print(fps)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print(fps)
