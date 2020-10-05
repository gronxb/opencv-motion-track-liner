# goodFeaturesToTrack 사용하지 않고 ROI로 특징점 직접 잡아줌
# 사진 5장 가량 활용해 이동 방향 나타낼 예정
import numpy as np
import cv2 as cv
img1 = cv.imread('1.jpg')          # queryImage
img2 = cv.imread('2.jpg') # trainImage

(x,y,w,h) = cv.selectROI('Select Window', img1, fromCenter = False, showCrosshair = True)

point_list = []
for _y in range(y,y+h,10):
    for _x in range(x,x+w,10):
        point_list.append((_x,_y))
points = np.array(point_list)
print(points.shape)
points = np.float32(points[:,np.newaxis,:])
print(points.shape)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
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

old_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

print(p0.shape)
mask = np.zeros_like(img1)


frame_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
good_new = p1[st==1]
good_old = points[st==1]
# draw the tracks

for i,(new,old) in enumerate(zip(good_new, good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    img2 = cv.circle(img2,(a,b),5,color[i].tolist(),-1)

img = cv.add(img2,mask)
cv.imshow('frame',img)
cv.waitKey(0)
