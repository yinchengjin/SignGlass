# echo-client.py

import socket
import time
import asyncio
import sys
import threading
import warnings
from picamera import PiCamera
from time import sleep
import os


warnings.filterwarnings("ignore", category=DeprecationWarning)


class TCPClient:
    def __init__(self):
        self.Host = "192.168.50.190"
        self.Port = 65432
        self.TCPClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.TCPClientSocket.connect((self.Host, self.Port))
        self.camera_inRecording = False
        self.camera = PiCamera()
        self.video_file_name = "-1"
        self.start_time = -1
        self.stop_time = -1
        self.file_index = '-1'
        self.user_name = 'Bob'
        self.file_folder = '-1'
        self.hostname = str(self.TCPClientSocket.getsockname()[0]).replace(".","")
        t1 = threading.Thread(target=self.recv_msg)
        t1.start()

        # t2 = threading.Thread(target=self.send_msg_sequential)
        # t2.start()

        # t3 = threading.Thread(target=self.send_msg)
        # t3.start()
        # self.send_msg_sequential()

    def recv_msg(self):
        while True:
            recv_msg = self.TCPClientSocket.recv(1024)
            if not recv_msg:
                sys.exit(0)
            # print(type(recv_msg),recv_msg)
            recv_msg = recv_msg.decode()
            recv_msg = list(recv_msg.split(" "))
            self.decode_msg(recv_msg)

    def create_folder(self):

        self.file_folder = './'+self.hostname+"/"+self.user_name+'/'

        if not os.path.exists(self.file_folder):
            os.makedirs(self.file_folder)


    def decode_msg(self,message):
        if message[0] in ["s",'r']:
            print("[-] Command: START video")
            self.user_name = message[1]
            self.file_index = message[2]
            self.create_folder()
            self.picamera_start()
            self.camera_inRecording = True

        if message[0] == "e":
            print("[-] Command: STOP video")
            self.picamera_stop()
            self.camera_inRecording = False


    def send_msg(self):
        while True:
            send_msg = input(str("Enter message: "))
            send_msg = send_msg.encode()
            self.TCPClientSocket.send(send_msg)
            print("Message sent")

    def send_msg_sequential(self):
        while True:
            self.TCPClientSocket.send(b"hello from Client")
            print("message sent")
            time.sleep(5)

    def picamera_start(self):
        if self.camera_inRecording == False:
            self.camera.resolution = (720, 480)
            self.camera.framerate = 30
            self.camera.start_preview()
            self.start_time = time.time() 
            time_message = "START " + str(self.start_time)
            self.TCPClientSocket.send(bytes(time_message, 'utf-8'))
            self.video_file_name = self.hostname+"_"+self.user_name+"_"+self.file_index
            self.camera.start_recording(self.video_file_name+".h264")

    def picamera_stop(self):
        if self.camera_inRecording:
            self.camera.stop_recording()
            self.camera.stop_preview()
            self.stop_time = time.time() 
            time_diff = self.stop_time - self.start_time
            time_message = "STOP " + str(self.stop_time) + " Diff "+ str(time_diff)
            self.TCPClientSocket.send(bytes(time_message, 'utf-8'))

            self.convert_h264_to_mp4()

    def convert_h264_to_mp4(self):
        try:
            command = "rm -r " + self.file_folder+ self.video_file_name+".mp4"
            os.system(command)  # delete mp4
        except:
            pass

        command = "MP4Box -add "+ self.video_file_name+'.h264' +" -fps 30 "+self.file_folder+ self.video_file_name+".mp4"
        os.system(command) #convert h264 to mp4

        command = "rm -r " + self.video_file_name + '.h264'
        os.system(command) # delete h264
        message = "Cam Video Stored "+ self.file_folder + self.video_file_name +".mp4"
        self.TCPClientSocket.send(bytes(message, 'utf-8'))
        print(command)

if __name__ == "__main__":
    TCPClient1 = TCPClient()