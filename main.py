#!/usr/bin/env python
#  alter for servos change /home/walter/maestro-linux/UscCmd to the location on your hard drive
# you may also need to change location of your camera
# Usage:
# 1. Install Python dependencies: cv2, flask, Coral AI, maestro code with if you want to hook up servos to move
# 2. Use Python 3.9 Coral won't work with later versions of python I used pyenv to use this version
# 3. Run "python3 main.py".
# 4. Navigate the browser to the local webpage. 
from flask import Flask, render_template, Response
import os
import sys
import threading
import argparse
import datetime
import imutils
import time
import cv2
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

tiltpos = 5000;
panpos = 5000;
maestrodir = '/home/walter/maestro-linux/UscCmd';

app = Flask(__name__)
def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/up')
def up():
    #up key button
    print ("up")
    global tiltpos
    tiltpos += 200;
    print (tiltpos);
    print(maestrodir + " --servo 0," + str(tiltpos))
    os.system(maestrodir + " --servo 0," + str(tiltpos))
    return ""
@app.route('/down') 
def down():
    #down key button
    print ("down")
    global tiltpos
    tiltpos -= 200;
    print (tiltpos);
    os.system(maestrodir + " --servo 0," + str(tiltpos))
    return ""
@app.route('/left')
def left():
    #left key button
    print ("left")
    global panpos
    panpos += 200;
    os.system(maestrodir + " --servo 1," + str(panpos))
    return ""
@app.route('/right')    
def right():
    #right key button
    print ("right")
    global panpos
    panpos -= 200;
    os.system(maestrodir + " --servo 1," + str(panpos))
    return ""    

def gen(camera):
    while True:   	
        success, image = camera.video.read()
        cv2_im = image

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, .4)
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

        frame = cv2.imencode('.jpg', cv2_im)
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #initialize servos
    os.system(maestrodir + " --servo 0," + str(tiltpos))
    os.system(maestrodir + " --servo 1," + str(panpos))
    interpreter = make_interpreter('./models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')

    labels = read_label_file('./models/coco_labels.txt')
    
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)
    
    app.run(host='0.0.0.0', debug=False)
    
