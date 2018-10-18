__author__ = 'suixin'

import numpy as np
import cv2
import pygame
from pygame.locals import *
import socket
import time
from carCtrHelper import RCTest

class CollectTrainingData(object):
    
    def __init__(self, host, port):

        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)

        # accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')
        # connect to a seral port
        print("准备初始化car")
        self.car = RCTest()
        pygame.init()
        pygame.display.set_mode((250, 250))
        self.send_inst = True

    def collect(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print("Start collecting images...")
        print("Press 'q' or 'x' to finish...")
        start = cv2.getTickCount()      

        # stream video frames one by one
        try:
            stream_bytes = b' '
            while self.send_inst:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')

                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    cv2.imshow('image', image)
                    total_frame += 1

                    # get input from human driver
                    for event in pygame.event.get():
                        if event.type == KEYDOWN:
                            key_input = pygame.key.get_pressed()

                         
                            if key_input[pygame.K_UP]:
                                saved_frame += 1
                                path ="./train_data/forward/" + str(int(time.time())) + ".jpg"
                                res = cv2.imwrite(path, image)
                                self.car.steer(0)
                                print("保存forward图片",res,path)
                            elif key_input[pygame.K_DOWN]:
                                print("Reverse")
                            elif key_input[pygame.K_LEFT]:
                                saved_frame += 1
                                path ="./train_data/left/" + str(int(time.time())) + ".jpg"
                                res = cv2.imwrite(path, image)
                                self.car.steer(1)
                                print("保存left图片",res,path)
                            elif key_input[pygame.K_RIGHT]:
                                saved_frame += 1   
                                path ="./train_data/right/" + str(int(time.time())) + ".jpg"
                                res = cv2.imwrite(path, image)
                                self.car.steer(2)
                                print("保存right图片",res,path)
                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print("exit")
                                self.send_inst = False                               
                                break                      
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break            

            end = cv2.getTickCount()
            # calculate streaming duration
            print("Streaming duration: , %.2fs" % ((end - start) / cv2.getTickFrequency()))
            print("Total frame: ", total_frame)
            print("Saved frame: ", saved_frame)
            print("Dropped frame: ", total_frame - saved_frame)
        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':
    # host, port
    h, p = "192.168.0.104", 8000
    ctd = CollectTrainingData(h, p)
    ctd.collect()
