# goodFeaturesToTrack 사용하지 않고 ROI로 특징점 직접 잡아줌
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('test.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

(x,y,w,h) = cv.selectROI('Select Window', old_frame, fromCenter = False, showCrosshair = True)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)


point_list = []
for _y in range(y,y+h,10):
    for _x in range(x,x+w,10):
        point_list.append((_x,_y))
points = np.array(point_list)
points = np.float32(points[:,np.newaxis,:])
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = points[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    points = good_new.reshape(-1,1,2)
