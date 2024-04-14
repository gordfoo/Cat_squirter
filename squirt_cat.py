import RPi.GPIO as GPIO
from time import sleep
import cv2

thres = 0.8 # Threshold to detect object

classNames = []
classFile = "/home/gordfoo/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/gordfoo/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/gordfoo/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=['cat']):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


# Set up GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.IN)
GPIO.setup(18, GPIO.OUT)
GPIO.setwarnings(False)

 


counter = 0

# Define a callback function to run when the input changes state
def callback(channel):
    print("Input detected on channel", channel)    
    GPIO.output(18, 1)
    sleep(4)
    GPIO.output(18, 0)
    sleep(4)
    
	

    

# Add the callback function to the input channel
GPIO.add_event_detect(24, GPIO.RISING, callback=callback)

# Wait for the input to change state
try:
    GPIO.wait_for_edge(24, GPIO.RISING)
except KeyboardInterrupt:
    GPIO.cleanup()
