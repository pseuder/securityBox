import threading
import time
import RPi.GPIO as GPIO
from bluepy.btle import Scanner, DefaultDelegate
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from flask import Flask, request
import picamera
from enum import Enum
import spidev
import struct
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
import tensorflow as tf
import argparse
import sys


GPIO.setmode(GPIO.BOARD)
BTN_PINS = 11
BUZZ_PIN = 16
MOTOR_PIN = 18
GPIO.setup(BTN_PINS, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUZZ_PIN, GPIO.OUT)
GPIO.setup(MOTOR_PIN, GPIO.OUT)

lock = True
move = False
human = False
alerm = False
motor_angle = 90

app = Flask(__name__)
@app.route('/cmd/<int:num>')
def cmd(num):
    global lock
    if num == 0:
        lock = True
        print("line bot 上鎖")
    elif num == 1:
        lock = False
        print("line bot 解鎖")


def serv():
    app.run(host='0.0.0.0', port=8000)

import requests
import random


#iot switch start
def switch(cmd):
    global lock
    if cmd == 1 and lock == False:
        lock = True
        print("iot lock")
    elif cmd == 0 and lock == True:
        lock = False
        print("iot unlock")

def get_var():
    try:               
        attempts = 0
        status_code = 400
        while status_code >= 400 and attempts < 5:            
            req = requests.get(url=URL, headers=HEADERS)
            status_code = req.status_code
            attempts += 1        
        # print(req.text)
        switch(int(float(req.text)))

    except Exception as e:
        print("[ERROR] Error posting, details: {}".format(e))

def iotSwitch():
    ENDPOINT = "things.ubidots.com"
    DEVICE_LABEL = "pi"
    VARIABLE_LABEL = "switch"
    TOKEN = "BBFF-emJqTmAM1pSYUweoLyisiiVXpAVFJy" # replace with your TOKEN
    DELAY = 0  # Delay in seconds
    global URL
    URL = "http://{}/api/v1.6/devices/{}/{}/lv".format(ENDPOINT, DEVICE_LABEL, VARIABLE_LABEL)
    global HEADERS
    HEADERS = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    try:
        while True:
            get_var()
            time.sleep(DELAY)
    except KeyboardInterrupt:
        print("Exception: KeyboardInterrupt")

    #iot switch end

def buttonPress(btn):
    global lock
    lock = not lock
    print("button press, lock = %r" % lock)
GPIO.add_event_detect(BTN_PINS, GPIO.FALLING, buttonPress, 200)


def detectMove():
    global move
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.mode = 0b11
    spi.max_speed_hz = 2000000

    def writeByte(reg, val):
        spi.xfer2([reg, val])

    def readByte(reg):
        packet = [0] * 2
        packet[0] = reg | 0x80
        reply = spi.xfer2(packet)
        return reply[1]
    writeByte(0x2D, 0x00)
    time.sleep(0.1)
    writeByte(0x2D, 0x08)
    writeByte(0x31, 0x08)
    time.sleep(0.5)
    while True:
        accel = {'x' : 0, 'y' : 0, 'z': 0}
        xAccl = struct.unpack('<h', bytes([readByte(0x32), readByte(0x33)]))[0]
        accel['x'] = xAccl / 256
        Moving_range = 0.3
        if accel['x'] > Moving_range or accel['x'] < -Moving_range :
            print("object moving!!")
            move = True
        else:
            move = False
        time.sleep(0.2)


IM_WIDTH = 640
IM_HEIGHT = 480

camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera', action='store_true')
args = parser.parse_args()
if args.usbcam: camera_type = 'usb'
sys.path.append('..')
from utils import label_map_util
from utils import visualization_utils as vis_util
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
camera = picamera.PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)



def detectHuman():
    global human
    for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: frame_expanded})
        cs, sc = np.squeeze(classes).astype(np.int32), np.squeeze(scores)
        # sc = np.squeeze(scores)
        for i in range(int(num[0])):
            if cs[i] == 1 and sc[i] > 0.7:
                print("human warning ")
                human = True
            else:
                human = False
        rawCapture.truncate(0)


def judgeSteal():
    global human, lock, move, alerm
    while True:
        if human == True and lock == True and move == True:
            print("human: %r" % human)
            print("物品被盜")
            alerm = True
            sendEmail()
            time.sleep(10)


pwm_buzzer = GPIO.PWM(BUZZ_PIN, 100)
pwm_buzzer.start(0)
def delay(times):
    time.sleep(times/1000.0)
    
def alerming():
    global alerm
    while True:
        if alerm == True:
            pwm_buzzer.ChangeDutyCycle(50)
            time.sleep(1)
            pwm_buzzer.ChangeDutyCycle(100)
            time.sleep(1)


def dealerming():
    global lock, alerm, move, human
    while True:
        if lock == False:
            alerm = False
            move = False
            human = False
            pwm_buzzer.ChangeDutyCycle(0)


def sendEmail():
    camera.capture("email.jpg")
    content = MIMEMultipart()  # 建立MIMEMultipart物件
    content["subject"] = "物品被盜"  # 郵件標題
    content["from"] = "iop890520@gmail.com"  # 寄件者
    content["to"] = "iop890520@gmail.com"  # 收件者
    content.attach(MIMEImage(Path("email.jpg").read_bytes()))  # 郵件圖片內容

    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        smtp.ehlo()  # 驗證SMTP伺服器
        smtp.starttls()  # 建立加密傳輸
        smtp.login("iop890520@gmail.com", "mwteidzxhsqzeprr")  # 登入寄件者gmail
        smtp.send_message(content)  # 寄送郵件
        print("Mail delivery complete!")


class ScanDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)

def phongScaning():
    global lock
    while True:
        scanner = Scanner().withDelegate(ScanDelegate())
        devices = scanner.scan(1.0)
        for dev in devices:
            if dev.addr == "d0:c5:f3:24:2f:31":
                if int(dev.rssi) > -33:
                    lock = False
                    print("手機解鎖")



pwm_motor = GPIO.PWM(MOTOR_PIN, 50)
pwm_motor.start(0)
def motoring():
    global lock, motor_angle
    while True:
        if lock == False:
            if motor_angle != 90:
                motor_angle = 90
                SetAngle(motor_angle)
        else:
            if motor_angle != 0:
                motor_angle = 0
                SetAngle(motor_angle)


def SetAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(18, True)
    pwm_motor.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(18, False)
    pwm_motor.ChangeDutyCycle(0)



_ser = threading.Thread(target=serv,)
_buttonPress = threading.Thread(target=buttonPress,)
_detectMove = threading.Thread(target=detectMove,)
_detectHuman = threading.Thread(target=detectHuman,)
_judgeSteal = threading.Thread(target=judgeSteal,)
_alerming = threading.Thread(target=alerming,)
_dealerming = threading.Thread(target=dealerming,)
_phongScaning = threading.Thread(target=phongScaning, )
_motoring = threading.Thread(target=motoring,)
_iotSwitch = threading.Thread(target=iotSwitch,)

try:

    _ser.start()
    _buttonPress.start()
    _detectMove.start()
    _detectHuman.start()
    _judgeSteal.start()
    _alerming.start()
    _dealerming.start()
    _phongScaning.start()
    _motoring.start()
    _iotSwitch.start()
except KeyboardInterrupt:
    pwm_buzzer.stop()
    pwm_motor.stop()
    spi.close()
    GPIO.cleanup()
    motor.cleanup()
    camera.close()
    cv2.destroyAllWindows()


