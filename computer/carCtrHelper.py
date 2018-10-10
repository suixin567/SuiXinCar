import socket
import struct

client = None

class RCTest(object):

    def __init__(self):
        global client        
        h, p1= "192.168.0.104", 8003
        self.client = Client(h,p1)
        print("helper init...")      

    def steer(self, prediction):
                    #print(prediction)
                    # simple orders
                    if prediction==2:
                        #print("Forward")
                        self.client.send("Forward")  
                    elif prediction==0:
                        #print("Left")
                        self.client.send("Left")
                    elif prediction==1:
                        #print("Right")
                        self.client.send("Right")                 
                    else:
                        #print("Stop")
                        self.client.send("Stop")
    def exit(self):
        self.client.close()
        
class Client(object):
    #global client
    def __init__(self, host, port1):
        self.host = host
        self.port1 = port1
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port1))
    def send(self,msg):
        #print("send")
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
    def close(self):
        self.client.close()
           