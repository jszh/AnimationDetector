import cv2
import imutils
import numpy as np
import socket
import sys
import time
import os
from transition import MovementDetector

HOST = '127.0.0.1'
BUF_SIZE = 4096

def getInt(bytes):
    return int.from_bytes(bytes, byteorder='little')

port_device = 1313
port_local = 12340
if len(sys.argv) == 3:
    port_device = int(sys.argv[1])
    port_local = int(sys.argv[2])
port_remote = port_local + 1

os.system('touch screenshot.jpg')
os.system('adb forward tcp:' + str(port_device) + ' localabstract:minicap')

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, port_device))
    s.setblocking(False)

    s_ana = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_ana.bind((HOST, port_local))
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
flag_imdecode_finished = False
detector = MovementDetector(should_draw=True)

last_mov_ts = 0
start_ts = 0
is_moving = False
is_blank = False
time_lim = 0.5

did_init_trans = False
trans_init_ts = 0

def save_image(path = 'screenshot.jpg'):
    global image
    image = imutils.resize(image, width=500)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
def handle_trans_fin():
    global is_moving, did_init_trans, last_mov_ts, start_ts, imgData
    is_moving = False
    if did_init_trans:
        try:
            save_image()
            # tell Analyzer about finishing after saving the screenshot
            s_ana.sendto(b'1', (HOST, port_remote))
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
            ti_str = trans_init.decode()
            if ti_str[0] == '1':
                img_path = ti_str[1:]
                start = time.time()
                while not flag_imdecode_finished:
                    if time.time() - start > 1:
                        print('wait decode finish timeout')
                print('wait time ', time.time() - start) 
                save_image(img_path)
                s_ana.sendto(b'2', (HOST, port_remote))
                continue
            elif ti_str == 'to_change':
                print('triggered')
                did_init_trans = True
                trans_init_ts = time.time()
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
        current_time = time.time()
        time_delta = current_time - last_mov_ts
        if time_delta > time_lim and is_moving:
            handle_trans_fin()
        if did_init_trans:
            trans_duration = current_time - trans_init_ts
            if (trans_duration > 3.0 and not is_moving) or (trans_duration > 10.0 and is_moving):
                print('abort', trans_duration)
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
                    flag_imdecode_finished = False
                    image = np.asarray(imgData, dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    flag_imdecode_finished = True
                    current_time = time.time()
                    curr_is_moving, is_blank = detector.process(image, orient, timestamp/1000, top_ignore=36)
                    time_lim = 5 if is_blank else 0.5

                    if curr_is_moving:
                        # current_time - last_mov_ts > time_lim and 
                        if not is_moving:
                            print('start')
                            is_moving = True
                            start_ts = current_time
                        last_mov_ts = current_time
                    else:
                        if current_time - last_mov_ts > time_lim and is_moving:
                            handle_trans_fin()
                        # cv2.imwrite('screenshot'+str(time.time())+'.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if did_init_trans:
                            trans_duration = current_time - trans_init_ts
                            if (trans_duration > 3.0 and not is_moving) or (trans_duration > 10.0 and is_moving):
                                print('abort', trans_duration)
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

