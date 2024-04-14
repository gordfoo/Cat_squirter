import RPi.GPIO as GPIO
from time import sleep
import cv2

thres = 0.5 # Threshold to detect object

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


import timeit
import RPi.GPIO as GPIO
from time import sleep

number = 18
print(number)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.OUT)
GPIO.setup(24, GPIO.IN)



#while (True):
#	GPIO.output(18, 0)
#	sleep(0.1)
#	GPIO.output(18, 1)
#	sleep(0.1)

def callback(channel):
    print("Input detected on channel", channel)
    
def trigger_camera():
    cap = cv2.VideoCapture(0)
    #cap.set(3,640) #Temporarily disable for performance
    #cap.set(4,480) #Temporarily disable for performance
    
    j = 400
    
    while j>0:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        print(objectInfo)
        if len(objectInfo) > 0:
            cv2.imwrite('result.png', result)
            trigger_squirter(cap)
            break
        #cv2.imshow("Output",img) #Temporarily disable for performance
        cv2.waitKey(100)
        j -= 100
    
    cap.release()
    cv2.destroyAllWindows()



def trigger_squirter(cap):    
    return_value, image = cap.read()
    #cv2.imwrite('opencvcat_before.png', image)
    GPIO.output(14, 1)
    sleep(0.1)
    #cv2.imwrite('opencvcat_after.png', image)
    GPIO.output(14, 0)
    #cv2.imwrite('opencvcat_after2.png', image)
    sleep(3)
    
while (True):
    if GPIO.input(24):
        trigger_camera()
        callback(24)
    sleep(0.5)
    print('just checked, nope')
    
