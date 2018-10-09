import socket
import struct
import pygame
from pygame.locals import *

client = None

class RCTest(object):

    def __init__(self):
        pygame.init()
        pygame.display.set_mode((250, 250))
        self.send_inst = True
        self.steer()

    def steer(self):

        while self.send_inst:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    key_input = pygame.key.get_pressed()

                    # complex orders
                    if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                        print("Forward Right")
                        client.send("Forward Right")
                    elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                        print("Forward Left")
                        client.send("Forward Left")
                    elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                        print("Reverse Right")
                        client.send("Reverse Right")
                    elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                        print("Reverse Left")
                        client.send("Reverse Left")
                    # simple orders
                    elif key_input[pygame.K_UP]:
                        print("Forward")
                        client.send("Forward")
                    elif key_input[pygame.K_DOWN]:
                        print("Reverse")
                        client.send("Reverse")
                    elif key_input[pygame.K_RIGHT]:
                        print("Right")
                        client.send("Right")
                    elif key_input[pygame.K_LEFT]:
                        print("Left")
                        client.send("Left")
                    # exit
                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print("Exit")
                        self.send_inst = False
                        client.send("Exit")
                        client.close()
                        break

                elif event.type == pygame.KEYUP:
                    print("up")
                    client.send("Stop")
class Client(object):
    #global client
    def __init__(self, host, port1):
        self.host = host
        self.port1 = port1
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port1))
    def send(self,msg):
        print("send")
        self.client.send(msg.encode('GBK'))
    # 第一步：先收报头
        header = self.client.recv(4)
    # 第二步：从报头中解析出对真实数据的描述信息（数据的长度）
        total_size = struct.unpack('i',header)[0]
    #print('total_size',total_size)
    # 第三步：接收真实的数据
        recv_size = 0
        recv_data = b''
        while recv_size < total_size:
            data = self.client.recv(1024) # 接收数据
            recv_data += data
            recv_size += len(data)   # 不能加1024，如果加进度条，会计算有误
        #print('resive', recv_data.decode('gbk', 'ignore')) 
    def close():
        self.client.close()
           
        

  
if __name__ == '__main__':
    global client
    print("client start...")
    h, p1= "192.168.0.104", 8003
    client = Client(h,p1)
    RCTest()
   # RCTest()