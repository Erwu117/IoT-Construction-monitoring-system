# Importing Libraries
from pathlib import Path
from datetime import datetime
import blobconverter
import time
import cv2
import depthai as dai
import numpy as np
from time import monotonic
from time import sleep
import itertools
from depthai_sdk import PipelineManager, NNetManager, PreviewManager
from depthai_sdk import cropToAspectRatio
import sys
import board
import adafruit_dht
import psutil
dhtDevice = adafruit_dht.DHT11(board.D4,use_pulseio=False)
import Adafruit_ADS1x15
import serial
import RPi.GPIO as GPIO
from MovenetRenderer import MovenetRenderer
from MovenetDepthai import MovenetDepthai

pose = MovenetDepthai(
    input_src="rgb",
    model="thunder",
    score_thresh=0.4,
    internal_frame_height=640
)

renderer = MovenetRenderer(pose)
i = -1
dustp = 0;
ledp = 23;
samp = 280;
delta = 40;
sleep = 9680;
vo_out = 0.0;
svolt = 0.0;
dL = 0.0;
incidents = 0;
GPIO.setmode(GPIO.BCM)
GPIO.setup(ledp, GPIO.OUT)
adc = Adafruit_ADS1x15.ADS1115()
sendnumber = "+639260729704"
ser = serial.Serial("/dev/serial0", 115200)
# GSM Function Code
def getresponse(sleep = 1):
    time.sleep(sleep)
    ret = ser.read_all()
    if not ret:
        print("ERR <Response Timeout>")
        return None
    ret = ret.decode()
    print("[RX]: ", ret)
    return ret
def commgsm(com):
    com = b"AT" + com.encode("UTF-8") + b"\r\n"
    print("[TX]: ", com.decode())
    ser.write(com)
def sendsms(msg):
    msg = msg + "\x1A"
    ser.write(msg.encode())
def initgsm():
    commgsm("E0")
    getresponse()
    commgsm("+CMEE=2")
    getresponse()
    commgsm("+CPIN?")
    getresponse()
    commgsm("+CLIP=1")
    getresponse()
    commgsm("+CMGF=1")
    getresponse()
# Alert notification functions
def tempalert():
    print(dt)
    commgsm("+CMGS=\"%s\""%sendnumber)
    if getresponse(0.1) :
        sendsms("{}\nCRITICAL: Current temperature is above threshold!\nTemp: {:.1f} C  Humidity: {}%".format(dt,temperature_c,humidity))
    print("CRITICAL: Current temperature is above threshold!")
    print("==Temperature Info==")
    print("Temp: {:.1f} C    Humidity: {}% ".format(temperature_c, humidity))
    time.sleep(10)

def fallalert():
    print(dt)
    commgsm("+CMGS=\"%s\""%sendnumber)
    if getresponse(0.1) :
        sendsms("{}\nCRITICAL: Fall Detected!\nKindly check the status of your workers.".format(dt))
    print("Critical: Fall Detected! Kindly check your workers' safety.")
    time.sleep(10)
    
def dustalert():
    print(dt)
    commgsm("+CMGS=\"%s\""%sendnumber)
    if getresponse(0.1) :
        sendsms("{}\nCRITICAL: Current Dust density level is above threshold!\nDust Density Level {:.5f} ug/m^3".format(dt,dl))
    print("CRITICAL: Current Dust density level is above threshold!")
    print("==Dust Level Info==")
    print("dust {:.5f} ug/m^3".format(dL))
    time.sleep(10)
    
while True:
    try:
        # Time and Data Configuration
        now = datetime.now()
        dt = now.strftime("%d/%m/%Y %I:%M:%S %p")
        frame, body = pose.next_frame()

        # Temperature Sensor Configuration
        temperature_c = dhtDevice.temperature
        humidity = dhtDevice.humidity
        # Dust initialization and detection algorithm
        GPIO.output(ledp, GPIO.LOW)
        time.sleep(samp/1000000)
        vo_out = adc.read_adc(dustp, gain=1);
        time.sleep(delta/1000000)
        GPIO.output(ledp, GPIO.HIGH)
        time.sleep(sleep/1000000)
        svolt = vo_out * (4.096/32767.0);
        dL = .1-(0.17 * svolt)
        if dL >= 10.0:
            dustalert()
            incidents = incidents + 1
        print("dust {:.5f} ug/m^3".format(dL));
        # Run code once inside the loop function
        while i < 0:
            # Initialize SIM800L Module
            initgsm()
            commgsm("+CMGS=\"%s\""%sendnumber)
            if getresponse(0.1) :
                sendsms("%s\nINFO: System is running"%dt)
            print(dt)
            print("INFO: System is running")
            i = 1
        
        # Scheduled Notification for Workers' condition
        current_time =  datetime.now().hour
        minute_time = datetime.now().minute
        # Notify safety engineer on 9 AM, 11 AM, 2 PM, and 5 PM
        if (current_time == 9 and minute_time == 0) or (current_time == 11 and minute_time == 0) or (current_time == 14 and minute_time == 0) or (current_time == 17 and minute_time == 0):
            commgsm("+CMGS=\"%s\""%sendnumber)
            if getresponse(0.1) :
                sendsms("{}\nScheduled Notification\nNumber of Incidents: {}\nTemp: {:.1f} C  Humidity: {}%".format(dt,incidents,temperature_c,humidity))

        if frame is None:
            break
        
        if sum(body.scores > body.score_thresh) > 8:
            keypoints = np.clip(body.keypoints, [0,0], [frame.shape[1], frame.shape[0]])
            x, y, w, h = cv2.boundingRect(keypoints)
            I = np.zeros_like(frame,dtype=np.uint8)
            I = renderer.draw(I, body)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            I = np.clip(I, 0, 1) * 255
            I = pose.crop_and_resize(I, pose.crop_region)
            I = cv2.resize(I, (128,128))
            frame_ac = dai.NNData()
            frame_ac.setLayer("input", I.ravel())
            pose.q_ac_in.send(frame_ac)
            crown_proportion = w / h
            predict = pose.q_ac_out.get()
            predict = np.array(predict.getLayerFp16("output")).reshape(-1, 2)
            action_id = int(np.argmax(predict))
            rate = 0.6 * predict[:, action_id] + 0.4 * (crown_proportion - 1)
            
            if rate > 0.55:
                pose_detect = "fall"
                # print(predict)
                if rate > 0.7:
                   fallalert()
                   incidents = incidents + 1
                   print("{}\n".format(incidents))

                cv2.putText(
                        frame,
                        pose_detect,
                        (40,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0,255,0),
                        2,
                    )
                if rate > 1:
                    rate = 1
                fall_detect = rate
                normal_detect = 1 - rate
            else:
                pose_detect = "normal"
                if rate >= 0.5:
                    fall_detect = 1 - rate
                    normal_detect = rate
                else:
                    cv2.putText(
                        frame,
                        pose_detect,
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        2,
                    )
                    fall_detect = rate
                    normal_detect = 1 - rate
            cv2.imshow("", I)
        renderer.draw(frame, body)
        key = renderer.waitKey(delay=1)
        if key == ord('q'):
            break
        if temperature_c or humidity is None:
            continue
        if temperature_c > 40:
            tempalert()
            incidents = incidents + 1
        if temperature_c is None:
            continue
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print("Warning: ")
        print(error.args[0])
        time.sleep(0.3)
        continue
    sleep(0.5)   
renderer.exit()
pose.exit()