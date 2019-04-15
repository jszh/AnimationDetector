import cv2
import imutils
import numpy as np
import socket
import time
from transition import MovementDetector

HOST = '127.0.0.1'
PORT = 1313
PORT_SERVER = 12340
PORT_ANALYZER = 12341
BUF_SIZE = 4096

def getInt(bytes):
    return int.from_bytes(bytes, byteorder='little')

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.setblocking(False)

    s_ana = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_ana.bind((HOST, PORT_SERVER))
    s_ana.setblocking(False)
except socket.error as msg:
    s.close()
    s_ana.close()
    print('Bind failed. Error: ' + str(msg))
    exit(1)

connected = False
counter = 0
readingImg = False
imgSize = 0
imgData = bytearray()
image = None
data = bytearray()
leftover = bytearray()
counter = 0

detector = MovementDetector(should_draw=True)

last_mov_ts = 0
start_ts = 0
is_moving = False
is_blank = False
time_lim = 0.5

did_init_trans = False

def handle_trans_fin():
    global is_moving, did_init_trans, last_mov_ts, start_ts, imgData
    is_moving = False
    if did_init_trans:
        try:
            cv2.imwrite('screenshot.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # tell Analyzer about finishing after saving the screenshot
            s_ana.sendto(b'1', (HOST, PORT_ANALYZER))
        except Exception as e:
            print(e)
        did_init_trans = False
    current_time = time.time()
    print('timeout: %0.2f' % (current_time-last_mov_ts) + '\tduration: %0.2f' % (current_time-start_ts))

while True:
    # listen to Analyzer
    try:
        trans_init = s_ana.recv(BUF_SIZE)
        if len(trans_init) > 0:
            print('triggered')
            did_init_trans = True
            if is_moving:
                time_lim = max(2, time_lim)
            elif time_lim == 2:
                time_lim = 0.5
    except BlockingIOError:
        pass
    except Exception as e:
        print(e)

    # receive img from Android device
    try:
        data = s.recv(BUF_SIZE)
    except BlockingIOError:
        time.sleep(0.005)
        time_delta = time.time() - last_mov_ts
        if time_delta > time_lim and is_moving:
            handle_trans_fin()


    # append new data to the end of leftover old data
    data = leftover + data
    leftover = bytearray()
    # repeat until nothing is left in buffer
    while len(data) > 0:
        offset = 0  # pointer for current byte
        if not connected:   # read connection headers
            version = data[0]
            header_length = data[1]
            pid = getInt(data[2:6])
            width = getInt(data[6:10])
            height = getInt(data[10:14])
            v_width = getInt(data[14:18])
            v_height = getInt(data[18:22])
            orient = data[22]
            quirk = data[23]
            connected = True
            offset = header_length

            print(pid, width, height, v_width, v_height, orient, quirk)
        
        if not readingImg:
            # start reading image
            if len(data) > offset+12:
                imgSize = getInt(data[offset:offset+4])
                timestamp = getInt(data[offset+4:offset+12])
                # print(counter, imgSize/1000, timestamp/1000, sep='\t')
                offset += 12
                imgData = bytearray()
                readingImg = True
                counter += 1
            else:
                imgSize = 0

        if readingImg:
            if len(data) - offset <= imgSize:
                if len(data) - offset == imgSize:
                    readingImg = False
                imgData += data[offset:]
                imgSize -= len(data) - offset
                offset = len(data)
            elif len(data) - offset > imgSize:
                imgData += data[offset:imgSize+offset]
                offset += imgSize
                imgSize = 0
                readingImg = False

            if not readingImg:    # image read finish
                if counter % 2 == 0:
                    image = np.asarray(imgData, dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    current_time = time.time()
                    curr_is_moving, is_blank = detector.process(image, orient, timestamp/1000, top_ignore=36)
                    time_lim = 5 if is_blank else 0.5

                    if curr_is_moving:
                        if current_time - last_mov_ts > time_lim and not is_moving:
                            print('start')
                            is_moving = True
                            start_ts = current_time
                        last_mov_ts = current_time
                    else:
                        if current_time - last_mov_ts > time_lim and is_moving:
                            handle_trans_fin()
                    # scaled = imutils.resize(image, width=200)
                    # cv2.imshow('Screen Capture', scaled)
                    cv2.waitKey(1)

        # if (len(data)-offset <= 12) and offset < len(data):
        #     print(offset, len(data), imgSize)


        if offset < len(data):
            if len(data)-offset > 12:
                data = data[offset:]
            else:
                leftover = bytearray(data[offset:])
                data = bytearray()
        else:
            data = bytearray()

