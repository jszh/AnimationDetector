import cv2
import imutils
import math
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import metrics
import sys

COLLECT_DATA = True

cnts2 = []
frame = None
positive = []
vals = []
def click(event, x0, y0, flags, param):
    global cnts2, frame
    if event == cv2.EVENT_LBUTTONUP:
        for i, cnt in enumerate(cnts2):
            (x, y, w, h) = cv2.boundingRect(cnt)
            if x0 >= x and x0 <= x+w and y0 >= y and y0 <= y+h: # found
                if positive[i]:
                    col = (255, 0, 0)
                else:
                    col = (0, 255, 0)
                cv2.rectangle(frame, (x,y), (x+w,y+h), col, 1)
                cv2.imshow("Video", frame)
                positive[i] = not positive[i]

# load video and parse args
fname = "SVID_20190410_154042_1"
if len(sys.argv) > 1:
    fname = sys.argv[1]
rotate = False
if len(sys.argv) > 2:
    if sys.argv[2] == 'r':
        rotate = True
vs = cv2.VideoCapture("/Users/Jason/Desktop/Interface/Recordings/"+fname+".mp4")

# initialize the first frame in the video stream
lastFrame = None

grayscale_bins = list(range(257))

index = 0
cv2.namedWindow("Video")
if COLLECT_DATA:
    cv2.setMouseCallback("Video", click)

# loop over the frames of the video
_, frame = vs.read()

while True:
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    if key == ord("n"):
        if not lastFrame is None:
            if len(np.unique(positive)) == 2:
                for i, cnt in enumerate(cnts2):
                    (pos, var, w, h) = vals[i]
                    print(pos, var, w, h, 1 if positive[i] else 0, sep=',')

        positive = []
        vals = []
        _, frame = vs.read()
        if rotate:
            frame = imutils.rotate_bound(frame, 90)
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()
    elif key == ord("y"):
        if not lastFrame is None:
            if len(np.unique(positive)) >= 1:
                for i, cnt in enumerate(cnts2):
                    (pos, var, w, h) = vals[i]
                    print(pos, var, w, h, 1 if positive[i] else 0, sep=',')

        positive = []
        vals = []

        # grab the current frame 
        _, frame = vs.read()
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break
        if rotate:
            frame = imutils.rotate_bound(frame, 90)
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()
    else:
        cv2.imshow("Video", frame)
        continue
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray,20,50)
    # minLineLength = 50
    # maxLineGap = 20
    # edges = cv2.dilate(edges, None, iterations=1)
    # edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    # 
 
    # if the first frame is None, initialize it
    if lastFrame is None:
        lastFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(lastFrame, gray)

    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=4)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    cnts2 = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    positive = [False] * len(cnts2)
 
    # loop over the contours
    area = 0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 100:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    for c2 in cnts2:    # permanent contour
        # if the contour is too small, ignore it
        # if cv2.contourArea(c2) < 50:
        #     continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.fillPoly(edges, pts =[c2], color=(255,255,255))

        # use grayscale for color
        # add 2 pixels of padding to include more background color
        y0 = max(y-2, 0)
        y1 = min(gray.shape[0], y+h+2)
        x0 = max(x-2, 0)
        x1 = min(gray.shape[1], x+w+2)
        hist, _ = np.histogram(gray[y0:y1,x0:x1], bins=grayscale_bins)
        hist[::-1].sort()   # sort hist in reverse order
        if hist[0] == 0:
            continue
        pos = 1     # pos: cutoff at 5%
        while hist[pos]/hist[0] > 0.05:
            pos += 1
        variance = hist[5:].std()
        vals.append((pos, variance, w, h))
        # print(x,y,w,h,pos)
        cv2.putText(frame, str(pos) + ';' + "%.0f" % variance + ';' + str(w*h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1)

        # calculate overlap area between movement and visual elements
        # perc = 0.6
        # idx = -1
        # m_overlap_area = 0
        # for j, c1 in enumerate(cnts):
        #     (x1, y1, w1, h1) = cv2.boundingRect(c1)
        #     xo1 = max(x, x1)
        #     yo1 = max(y, y1)
        #     xo2 = min(x+w, x1+w1)
        #     yo2 = min(y+h, y1+h1)
        #     if xo1 < xo2 and yo1 < yo2: # if there is overlap
        #         overlap_area = (xo2-xo1) * (yo2-yo1)
        #         perc1 = min(overlap_area / cv2.contourArea(c1), 1)
        #         perc2 = min(overlap_area / cv2.contourArea(c2), 1)
        #         if math.sqrt(perc1 * perc2) > perc:
        #             perc = math.sqrt(perc1 * perc2)
        #             idx = j
        #             m_overlap_area = min(overlap_area, cv2.contourArea(c1), cv2.contourArea(c2))
        # if idx >= 0:
        #     # print(idx, perc)
        #     area += m_overlap_area
        #     (x1, y1, w1, h1) = cv2.boundingRect(cnts[idx])
        #     cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 1)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

    movement_perc = area/(frame.shape[0]*frame.shape[1])
    # print(area, movement_perc, len(cnts), len(cnts2))
    # print("Frame", index)
    index += 1

    # draw lines
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Movement ratio: {0:.2f}".format(movement_perc), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    # show the frame and record if the user presses a key
    # cv2.imshow("Threshold", thresh)
    # cv2.imshow("Edge", edges)
    cv2.imshow("Video", frame)
    # cv2.imshow("Delta", gray)
    



    lastFrame = gray
 
    

 
# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()
