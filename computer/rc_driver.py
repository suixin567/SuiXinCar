# coding=UTF-8
__author__ = 'zhengwang'

import sys
import threading
import socketserver
import numpy as np
import time

from model import NeuralNetwork
from rc_driver_helper import *
from carCtrHelper import RCTest

# distance data measured by ultrasonic sensor
sensor_data = None


class SensorDataHandler(socketserver.BaseRequestHandler):

    data = " "

    def handle(self):
        global sensor_data
        while self.data:
            self.data = self.request.recv(1024)
            sensor_data = round(float(self.data), 1)
            # print "{} sent:".format(self.client_address[0])
            print(sensor_data)


class VideoStreamHandler(socketserver.StreamRequestHandler):

    # h1: stop sign, measured manually
    # h2: traffic light, measured manually
    h1 = 5.5  # cm
    h2 = 5.5

    # load trained neural network
    nn = NeuralNetwork()
    nn.load_model("saved_model/nn_model.xml")

    obj_detection = ObjectDetection()
    rc_car = RCTest()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier("cascade_xml/stop_sign.xml")
    light_cascade = cv2.CascadeClassifier("cascade_xml/traffic_light.xml")

    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_light = 25

    stop_start = 0  # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0

    def handle(self):

        global sensor_data
        stream_bytes = b' '
        stop_flag = False
        stop_sign_active = True

        try:
            # stream video frames one by one
            tempCount=0
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    gray = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    #image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('image', gray)
                    # cv2.imshow('mlp_image', roi)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("car stopped")
                            self.rc_car.exit()
                            break       
                    tempCount=tempCount+1
                    #print(tempCount)
                    if tempCount>50:
                        tempCount=0
                        # lower half of the image
                        height, width = gray.shape
                        roi = gray[int(height/2):height, :]
                        # reshape image
                        image_array = roi.reshape(1, int(height/2) * width).astype(np.float32)
    
                        # neural network makes prediction
                        prediction = self.nn.predict(image_array)
                        print(prediction)
                        self.rc_car.steer(prediction)
                        self.stop_start = cv2.getTickCount()
                        self.d_stop_sign = 25
    
                        if stop_sign_active is False:
                                self.drive_time_after_stop = (self.stop_start - self.stop_finish) / cv2.getTickFrequency()
                                if self.drive_time_after_stop > 5:
                                    stop_sign_active = True
    
                       
        finally:
            cv2.destroyAllWindows()
            sys.exit()


class Server(object):
    def __init__(self, host, port1, port2):
        self.host = host
        self.port1 = port1
        self.port2 = port2

    def video_stream(self, host, port):
        s = socketserver.TCPServer((host, port), VideoStreamHandler)
        s.serve_forever()

    def sensor_stream(self, host, port):
        s = socketserver.TCPServer((host, port), SensorDataHandler)
        s.serve_forever()

    def start(self):
        sensor_thread = threading.Thread(target=self.sensor_stream, args=(self.host, self.port2))
        sensor_thread.daemon = True
        sensor_thread.start()
        self.video_stream(self.host, self.port1)


if __name__ == '__main__':
    h, p1, p2 = "192.168.0.103", 8000, 8002

    ts = Server(h, p1, p2)
    ts.start()
