__author__ = 'suixin'

import sys
import threading
import socketserver
import numpy as np
import cv2
from predict import predict2
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
    car = RCTest()
    count =0
    def handle(self):

        global sensor_data
        stream_bytes = b' '

        try:
            # stream video frames one by one
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    #gray = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('image', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("car stopped")
                        break
                    self.count = self.count+1
                    while self.count>50:
                        self.count = 0
                        #predict
                        direction  = predict2(image)                        
                        self.car.steer(direction)
                    
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
    h, p1, p2 = "192.168.0.104", 8000, 8002

    ts = Server(h, p1, p2)
    ts.start()
