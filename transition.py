import cv2
import imutils
import math
import numpy as np
from sklearn import metrics
from sklearn.cluster import MeanShift
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sys
from time import time
import pickle

DEBUG = False

class MovementDetector:

    def __check_overlap(self, x,y,w,h, x1,y1,w1,h1):
        xo1 = max(x, x1)
        yo1 = max(y, y1)
        xo2 = min(x+w, x1+w1)
        yo2 = min(y+h, y1+h1)
        if xo1 < xo2 and yo1 < yo2:
            return (xo2-xo1) * (yo2-yo1)
        return 0

    def __merge_rect(self, rects, flags, r0):
        assert(len(rects) == len(flags))
        (x,y,w,h) = r0 
        for i, r in enumerate(rects):
            if flags[i]:
                x1,y1,w1,h1 = r
                x = min(x, x1)
                y = min(y, y1)
                w = max(x+w, x1+w1) - x
                h = max(y+h, y1+h1) - y
        return (x,y,w,h)

    def __get_thresh(self, delta, thr=20):
        thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=3)
        thresh = cv2.erode(thresh, None, iterations=3)
        return thresh

    def __get_sorted_hist(self, img):
        hist, _ = np.histogram(img, bins=self.grayscale_bins)
        hist[::-1].sort()
        return hist

    def __init__(self, should_draw):
        # load GBDT model
        fin = open('save/gbdt.pickle', 'rb')
        self.clf = pickle.load(fin)
        # load data scaler
        fin = open('save/scaler.pickle', 'rb')
        self.scaler = pickle.load(fin)

        # initialize the first frame in the video stream
        self.lastFrame = None

        self.grayscale_bins = list(range(257))

        self.mov_history = []
        self.mov_ttl = []
        self.MAX_TTL = 25
        self.should_draw = should_draw

        self.diff_history = []
        self.time_history = []

        self.last_ts = 0
        self.DEFAULT_FPS = 60


    # returns (is_moving, is_blank)
    def process(self, frame, rotation, timestamp, top_ignore=75):
        # resize and rotate the frame
        if rotation == 0:
            frame = imutils.resize(frame, width=500)
        else:
            frame = imutils.resize(frame, height=500)
            frame = imutils.rotate_bound(frame, rotation)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,20,50)
    
        # if the first frame is None, initialize it
        if self.lastFrame is None:
            self.lastFrame = gray
            self.last_ts = timestamp
            return (False, False)

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(self.lastFrame, gray)
        thresh = self.__get_thresh(frameDelta)

        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        # cnts: contours for diff between two frames (motion)
        cnts = imutils.grab_contours(cnts)

        # cnts2: contours for edges in current frame (static)
        cnts2, hierarchy = cv2.findContours(edges, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)


        # convert elapsed time (sec) to TTL (60-fps-based)
        ttl_multiplier = int(round((timestamp - self.last_ts) / (1.0 / self.DEFAULT_FPS)))
        if ttl_multiplier == 0:
            ttl_multiplier += 1

        # all image regions
        images = []

        # prepare for movement detection
        is_moving = False
        # remove old entries
        self.mov_ttl = [t-ttl_multiplier for t in self.mov_ttl]
        self.mov_history = [cnt for k,cnt in enumerate(self.mov_history) if self.mov_ttl[k] > 0]
        self.mov_ttl = [t for t in self.mov_ttl if t > 0]
        assert(len(self.mov_history) == len(self.mov_ttl))
    
        area = 0

        # central region of screenshot: 25% from top and bottom, 20% from left and right
        is_blank = True
        xc1 = int(gray.shape[1] * 0.2)
        xc2 = gray.shape[1] - xc1
        yc1 = int(gray.shape[0] * 0.25)
        yc2 = gray.shape[0] - yc1

        for i, c2 in enumerate(cnts2):    # permanent contour
            # if the contour is too small, ignore it
            if cv2.contourArea(c2) < 10 or hierarchy[0][i][3] != -1:
                continue
            (x, y, w, h) = cv2.boundingRect(c2)
            if self.should_draw:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # if self.__check_overlap(x, y, w, h, xc1, yc1, xc2-xc1, yc2-yc1) > 0:
            #     is_blank = False

            # histogram approach
            # use grayscale for color
            # add 2 pixels of padding to include more background color
            y0 = max(y-2, 0)
            y1 = min(gray.shape[0], y+h+2)
            x0 = max(x-2, 0)
            x1 = min(gray.shape[1], x+w+2)
            hist = self.__get_sorted_hist(gray[y0:y1,x0:x1])
            if hist[0] == 0:
                continue
            pos = 1     # pos: cutoff at 5%, measures the diversity of color
            while hist[pos]/hist[0] > 0.05:
                pos += 1
                if pos == 255: break
            variance = hist[5:].std()
            scaled = self.scaler.transform([[pos, variance, w, h]])
            is_img = self.clf.predict(scaled)[0]
            # print(x,y,w,h,pos)
            if self.should_draw and DEBUG:
                cv2.putText(frame, str(pos) + ';' + "%.0f" % variance, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1)  # + ';' + str(w) + ';' +str(h)
            if is_img == 1:
                if w*h/(frame.shape[0]*frame.shape[1]) < 0.8:
                    images.append((x,y,w,h))

        # check for blank spaces
        min_mov_area = 20
        min_tot_area = 45
        if is_blank:
            hist = self.__get_sorted_hist(gray[yc1:yc2,xc1:xc2])
            if hist[0] != 0:
                # lower the criteria for movement
                if hist[1]/hist[0] < 0.005:
                    min_mov_area = 10
                    min_tot_area = 20
                else:
                    is_blank = False


        # merge overlapped image parts
        retain = [True] * len(images)
        i = 0
        while i < len(images):
            if not retain[i]:
                i += 1
                continue
            (x1,y1,w1,h1) = images[i]
            merge_flag = [False] * len(images)
            should_merge = False
            for j, r2 in enumerate(images):
                if i != j and retain[j]:
                    (x2,y2,w2,h2) = r2
                    overlap = self.__check_overlap(x1,y1,w1,h1, x2,y2,w2,h2)
                    if overlap > 0:
                        merge_flag[j] = True
                        retain[j] = False
                        should_merge = True
            if should_merge:
                retain[i] = False
                images.append(self.__merge_rect(images, merge_flag, images[i]))
                retain.append(True)
            i += 1

        images = [r for i,r in enumerate(images) if retain[i]]

        # remove the areas under each image from threshold graph
        # for use in the "second chance" part
        for rect in images:
            (x,y,w,h) = rect
            mask = np.zeros((h, w),np.uint8)
            thresh[y:y+h,x:x+w] = mask
            if self.should_draw:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # reduce TTL for disappeared images
        for i, hist_rect in enumerate(self.mov_history):
            (x2, y2, w2, h2) = hist_rect
            imgFound = False
            # search for bounding box in video contour history
            for img_rect in images:
                (x, y, w, h) = img_rect
                overlap = self.__check_overlap(x,y,w,h, x2,y2,w2,h2)
                if overlap / (w2*h2) > 0.9 and overlap / (w2*h2) < 1.1:   # found in current images
                    imgFound = True
                    break
            if not imgFound:
                self.mov_ttl[i] -= 5 * ttl_multiplier

        # detect moving area
        appended_to_tmp = [False] * len(images)
        tmp_mov_history = []
        tmp_mov_ttl = []
        total_area = 0
        image_moving = False
        for mv_cnt in cnts:     # for each movement area
            if cv2.contourArea(mv_cnt) < min_mov_area:
                continue
            (x1, y1, w1, h1) = cv2.boundingRect(mv_cnt)
            if x1 + w1 <= 160 and y1 + h1 <= top_ignore:   # ignore top status bar
                continue

            max_overlap = 0
            for i, img_rect in enumerate(images):   # for each image are
                (x, y, w, h) = img_rect
                overlap = self.__check_overlap(x-3,y-3,w+6,h+6, x1,y1,w1,h1)
                histFound = False
                if overlap > 0:     # movement area overlaps an image
                    # search for bounding box in video contour history
                    for j, hist_rect in enumerate(self.mov_history):
                        (x2, y2, w2, h2) = hist_rect
                        h_overlap = self.__check_overlap(x,y,w,h, x2,y2,w2,h2)
                        if h_overlap / (w2*h2) > 0.9 and h_overlap / (w2*h2) < 1.1:   # consider this a video
                            self.mov_ttl[j] = self.MAX_TTL
                            histFound = True
                            if self.should_draw:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 2)
                            break
                    if not histFound:   # potentially a new video
                        if not appended_to_tmp[i]:
                            tmp_mov_history.append(img_rect)
                            tmp_mov_ttl.append(self.MAX_TTL)
                            appended_to_tmp[i] = True
                        image_moving = True
                if overlap / (w1*h1) > max_overlap and histFound:   # update max_overlap
                    max_overlap = overlap / (w1*h1)
            if max_overlap < 0.8:
                whv = w1/h1
                if whv > 1: whv = 1/whv
                total_area += w1*h1 * (1 - max_overlap) * whv

        if total_area >= min_tot_area or image_moving:
            is_moving = True
            self.diff_history = []
            self.time_history = []

        # second chance: combine several prev diffs to see if there they form any regions
        elif not image_moving:
            self.diff_history = [dh for i,dh in enumerate(self.diff_history) if timestamp-self.time_history[i]<1]
            self.time_history = [t for t in self.time_history if timestamp-t<1]
            self.diff_history = self.diff_history[-12:]
            self.time_history = self.time_history[-12:]

            if len(self.diff_history) > 0:
                deltas = thresh.copy()
                for diff in self.diff_history:
                    deltas = cv2.add(deltas, diff)
                merged_thresh = self.__get_thresh(deltas)
                merged_cnts = cv2.findContours(merged_thresh, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                merged_cnts = imutils.grab_contours(merged_cnts)
                if self.should_draw and DEBUG:
                    cv2.imshow("Merged Threshold", merged_thresh)
                total_area = 0
                for mv_cnt2 in merged_cnts:
                    if cv2.contourArea(mv_cnt2) < min_mov_area:
                        continue
                    (x1, y1, w1, h1) = cv2.boundingRect(mv_cnt2)
                    if x1 + w1 <= 160 and y1 + h1 <= top_ignore:   # ignore top status bar
                        continue

                    max_overlap = 0
                    for i, c2 in enumerate(cnts2):      # for each image contour
                        if cv2.contourArea(c2) < 30 or hierarchy[0][i][3] != -1:
                            continue
                        (x, y, w, h) = cv2.boundingRect(c2)
                        overlap = self.__check_overlap(x-3,y-3,w+6,h+6, x1,y1,w1,h1)
                        if overlap > max_overlap:
                            max_overlap = overlap
                    whv = w1/h1
                    if whv > 1: whv = 1/whv
                    total_area += max_overlap * whv
                if total_area >= min_tot_area:
                    is_moving = True

            self.diff_history.append(thresh)
            self.time_history.append(timestamp)


        self.mov_history += tmp_mov_history
        self.mov_ttl += tmp_mov_ttl
        if DEBUG:
            print(self.mov_ttl, is_moving, total_area)
    
        # show the frame and record if the user presses a key
        if DEBUG and self.should_draw:
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Edge", edges)
        if self.should_draw:
            cv2.putText(frame, "Moving" if is_moving else "Stationary", (frame.shape[1]//2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Video", frame)
        # cv2.imshow("Delta", gray)
        
        self.lastFrame = gray
        self.last_ts = timestamp

        return (is_moving, is_blank)
 

def get_file():
    # load video and parse args
    fname = "SVID_20190410_154042_1"
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    rotate = False
    if len(sys.argv) > 2:
        if sys.argv[2] == 'r':
            rotate = True
    vs = cv2.VideoCapture("/Users/Jason/Desktop/Interface/Recordings/"+fname+".mp4")
    cv2.namedWindow("Video")
    return vs, rotate


if __name__ == '__main__':
    vs, rotate = get_file()
    start_time = time()
    detector = MovementDetector(should_draw=True)
    index = 0
    # loop over the frames of the video
    while True:
        # grab the current frame 
        _, frame = vs.read()
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break
        index += 1
        is_moving = detector.process(frame, 90 if rotate else 0, index * 0.025)

        timeout = 0 if is_moving else 1
        key = cv2.waitKey(timeout) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break


    print('Total run time:', time() - start_time)

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()
