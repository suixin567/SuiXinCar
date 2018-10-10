# coding=UTF-8
import socket
import struct
import RPi.GPIO as GPIO
import time

IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 22

SLEEP=0.05

def init():
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IN1,GPIO.OUT)
        GPIO.setup(IN2,GPIO.OUT)
        GPIO.setup(IN3,GPIO.OUT)
        GPIO.setup(IN4,GPIO.OUT)

def forward():
        GPIO.output(IN1,GPIO.HIGH)
        GPIO.output(IN2,GPIO.LOW)
        GPIO.output(IN3,GPIO.HIGH)
        GPIO.output(IN4,GPIO.LOW)
        time.sleep(SLEEP)
        stop()
        
def reverse():
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.HIGH)
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.HIGH)
        time.sleep(SLEEP)
        stop()
        
def left():
        GPIO.output(IN1,GPIO.HIGH)
        GPIO.output(IN2,GPIO.LOW)
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.LOW)
        time.sleep(SLEEP)
        stop()
        
def right():
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.LOW)
        GPIO.output(IN3,GPIO.HIGH)
        GPIO.output(IN4,GPIO.LOW)
        time.sleep(SLEEP)
        stop()

def pivot_left(tf):
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.HIGH)
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.LOW)
        time.sleep(tf)
        GPIO.cleanup()
# 
def pivot_right(tf):
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.LOW)
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.HIGH)
        time.sleep(tf)
        GPIO.cleanup()

# 
def p_left(tf):
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.HIGH)
        GPIO.output(IN3,GPIO.HIGH)
        GPIO.output(IN4,GPIO.LOW)
        time.sleep(tf)
        GPIO.cleanup()

# 
def p_right(tf):
        GPIO.output(IN1,GPIO.HIGH)
        GPIO.output(IN2,GPIO.LOW)
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.HIGH)
        time.sleep(tf)
        GPIO.cleanup()
        
        
def stop():
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.LOW)
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.LOW)
        
        
sock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_server.bind(('192.168.0.104', 8003))
sock_server.listen(0)# 开始监听，1代表在允许有一个连接排队，更多的新连接连进来时就会被拒绝
print('starting...')
init()
while True:
    conn, client_addr = sock_server.accept()
    print(client_addr)
    while True:
        try:
            data = conn.recv(1024)
            if not data: break# 适用于linux操作系统,防止客户端断开连接后死循环
            #sensor_data = float(data.decode('gbk'))
            command = data.decode('gbk')
            print(command)
            if command =="Forward":
                 forward()
            if command =="Reverse":
                 reverse()
            if command =="Left":
                 left()
            if command =="Right":
                 right()
            if command =="Stop":
                 stop()
            if command =="Exit":
                 stop()
                 #GPIO.cleanup()
                 break
            #响应
            res ="server".encode('GBK')
             # 第一步：制作固定长度的报头4bytes
            total_size = len(res)
            header = struct.pack('i',total_size)
             #  第二步：把报头发送给客户端
            conn.send(header)
             # 第三步：再发送真实的数据
            conn.sendall(res)            
        except ConnectionResetError:# 适用于windows操作系统,防止客户端断开连接后死循环
            break
    conn.close()
sock_server.close()