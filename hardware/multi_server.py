import socket
import sys
from _thread import start_new_thread
import threading
import time
import cv2
import read_file_name

HOST = '192.168.50.190' # all availabe interfaces
PORT = 65432 # arbitrary non privileged port
import keyboard

count = 0
message_sent_to_raspi = False
message_command = ""
last_command = '-1'
f = open("file_record.txt", "w")
f.close()
video_info = {"name": "macvideo", "path": "bob_47.mp4"}
parent_path = "./hardware/signglass_raspi"
file_info = []
# file_info_ready = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error as msg:
    print("Could not create socket. Error Code: ", str(msg[0]), "Error: ", msg[1])
    sys.exit(0)

print("[-] Socket Created")

# bind socket
try:
    s.bind((HOST, PORT))
    print("[-] Socket Bound to port " + str(PORT))
except socket.error as msg:
    print("Bind Failed. Error Code: {} Error: {}".format(str(msg[0]), msg[1]))
    sys.exit()

s.listen(10)
print("Listening...\n")

# The code below is what you're looking for ############

def client_thread(conn,addr):
    global message_command
    conn.send(b"Welcome to the Server. Type messages and press enter to send.\n")
    send_thread = threading.Thread(target=_helper_client_send_thread2, args=(conn,))
    send_thread.start()
    while True:
        data = conn.recv(1024)
        if not data:
            break
        reply = str(addr)+":"+str(data.decode('utf-8'))
        save_file_record(reply)
    conn.close()
    print("[-] DisConnected to " + str(addr[0]) + ": disconnected")

def _helper_client_send_thread2(conn):
    global count
    global message_sent_to_raspi
    global message_command
    global file_info
    while True:
        if message_sent_to_raspi == True:
            message_to_raspi = message_command + " " + ' '.join([str(i) for i in file_info])
            conn.sendall(message_to_raspi.encode("utf-8"))
            time.sleep(1)
            message_sent_to_raspi = False

def _helper_main_thread_keyboard2():
    global message_sent_to_raspi
    global message_command
    global last_command
    global file_info

    while True:
        key_press = keyboard.read_key()

        if last_command in ['s','r']:
            if key_press == 'e':
                file_sentence = "[+] Command: STOP Recording"
                save_file_record(file_sentence)
                message_command = key_press
                last_command = message_command
                message_sent_to_raspi = True
                time.sleep(0.5)

        if last_command in ['e','-1']:
            if key_press in ['s','r']:
                if key_press == 's':
                    file_info = read_file_name.get_list_of_file_name(parent_path)
                    save_file_record(str(file_info))
                    file_sentence = "[-] Command: START Recording"
                    save_file_record(file_sentence)
                if key_press == 'r':
                    file_info = read_file_name.get_list_of_file_name(parent_path,1)
                    save_file_record(str(file_info))
                    file_sentence = "[-] Command: ReSTART Recording"
                    save_file_record(file_sentence)

                message_command = key_press
                last_command = message_command
                message_sent_to_raspi = True


        if last_command == 'z':
            print("keyboard while stop")
            break

def make_1080p(video):
    video.set(3, 1920)
    video.set(4, 1080)

def captrure_video_keyboard(video_path):
    global message_command

    video = cv2.VideoCapture(0)

    if (video.isOpened() == False):
        print("Error reading video file")

    make_1080p(video)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))


    size = (frame_width, frame_height)
    result = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc('m', 'ps', '4', 'v'),30, size)
    start_time = time.perf_counter()
    list_of_frame = []

    start = False
    try:
        while True:
            if message_command in ['s','r']:
                ret, frame = video.read()
                if start == False:
                    start_time = time.time()
                    file_record_message = "(local): START "+str(start_time)
                    save_file_record(file_record_message)
                start = True
                frame2 = cv2.flip(frame, 1)
                list_of_frame.append(frame2)

                if message_command == 'e':
                    end_time = time.time()
                    file_record_message ="(local): STOP "+str(end_time)+" Diff "+str(end_time-start_time)
                    save_file_record(file_record_message)
                    break

    except KeyboardInterrupt:
        pass

    # print(list_of_frame)
    for fram in list_of_frame:
        result.write(fram)

    video.release()
    result.release()
    cv2.destroyAllWindows()
    file_record_message = "(local): Cam Video Stored " + video_path
    save_file_record(file_record_message)

def save_file_record(file_sentence):
    tmp_f = open("file_record.txt", "a")

    print(file_sentence)
    tmp_f.write(file_sentence + '\n')
    tmp_f.close()

def video_capture_keyboard():
    global message_command
    global file_info


    while True:
        if message_command in ['s','r']:
            video_path = parent_path+"/experiment result/local/"+file_info[0]+"/local_"+ file_info[0] + "_" + file_info[1] + ".mp4"
            captrure_video_keyboard(video_path)
        if message_command == 'z':
            print("video capture while stoped")
            break

def main():

    try:
        keyboard_detect_thread = threading.Thread(target=_helper_main_thread_keyboard2)
        keyboard_detect_thread.start()

        video_capture_thread = threading.Thread(target=video_capture_keyboard)
        video_capture_thread.start()

        while True:
            # blocking call, waits to accept a connection
            conn, addr = s.accept()
            print("[-] Connected to " + str(addr[0]) + ":"+ str(addr) )

            start_new_thread(client_thread, (conn,addr,))

    except:
        s.close()

if __name__ == "__main__":
    main()
