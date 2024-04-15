import RPi.GPIO as GPIO
from time import sleep
import cv2
import datetime
import RPi.GPIO as GPIO
import time
import threading

THRES = 0.5 # Threshold to detect object

SPEAKER_PIN = 21
SQUIRTER_PIN = 14
IR_SENSOR_PIN = 24

number = 18

today = datetime.date.today()

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SPEAKER_PIN, GPIO.OUT)
GPIO.setup(SQUIRTER_PIN, GPIO.OUT)
GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
GPIO.setmode(GPIO.BCM)

classNames = []
classFile = "coco.names"

with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=['cat']):
    print(img.shape)
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
    
def trigger_camera():
    """Trigger the camera connected to the computer. Opens the video capture
    and passes the image to cap.read() based on a preset interval and number
    of cycles. If the image returned matches a cat, then the external devices
    are activated (e.g. trigger_squirter())
    """
     
    cap = cv2.VideoCapture(0)
    #cap.set(3,640) #Temporarily disable for performance
    #cap.set(4,480) #Temporarily disable for performance
    
    #How many cycles to do    
    cycles = 4
    
    #How long to wait between each cycles
    wait_time = 100
    
    #Loop the number of times equal to the cycles
    for _ in range(cycles):
        
        #Gather result from cap.read()
        success, img = cap.read()
        
        #Call getObjects() to get result of image processing
        result, objectInfo = getObjects(img,0.45,0.2)
        print(objectInfo)
        
        #If there is information about an object, the length of this list will
        #be non-zero
        if len(objectInfo) > 0:
            cv2.imwrite('result.png', result)
            activate_external_devices(cap)
            take_face_shot(cap)
            break
        #Wait to process next image
        cv2.waitKey(wait_time)
        
    
    cap.release()
    cv2.destroyAllWindows()
    
def take_face_shot(cap):
    #Smile for the camera, kitty!
    
    today = today()
    
    num_pics = 4
    
    while num_pics > 0:
        #Read the image from the video feed
        _result, img = cap.read()
        
        #Write the image to the file
        filename = f'funny_cat_pic_{num_pics}_{today}.png'
        
        cv2.imwrite(filename, img)
    
    
def trigger_squirter():    
   #Length of squirt in seconds
    
    #Wait 1 second, to allow the cat to hear the beep first
    time.sleep(1)
    
    #Length of water squirt
    squirt_length = 0.2
    
    #Send signal through the GPIO pin to activate
    GPIO.output(SQUIRTER_PIN, 1)
    
    #Leave the GPIO activated for a time equivalent to squirt_length
    sleep(squirt_length)

    #Put squirter back to 0
    GPIO.output(SQUIRTER_PIN, 0)

    
    
    
# Function to generate a tone
def trigger_speaker():
    
    frequency = 800
    duration = 2
    pulse_size = 4
    duty_cycle_percentage = 50 #Has to do with waveform. 50 usually works best
    
    pulse_time = 1 / pulse_size
    pulses = round(duration / pulse_time)
    
    pwm = GPIO.PWM(SPEAKER_PIN, frequency)
    
    for _ in range(pulses):
        pwm.start(duty_cycle_percentage)
        time.sleep(pulse_time)
        pwm.stop()
        time.sleep(pulse_time)
        

    # Cleanup GPIO



def activate_external_devices(cap):
    # Create threads for each function
    thread_squirter = threading.Thread(target=trigger_squirter)
    thread_speaker = threading.Thread(target=trigger_speaker)
    thread_camera = threading.Thread(target=take_face_shot, args=(cap,))

    # Start all threads
    thread_squirter.start()
    thread_speaker.start()
    thread_camera.start()

    # Wait for all threads to finish
    thread_squirter.join()
    thread_speaker.join()
    thread_camera.join()


    print("All tasks completed")

if __name__ == "__main__":
    while (True):
        if GPIO.input(24):
            trigger_camera()
        current_time = time.time()
        print(f'Check occured at {current_time}')
        sleep(0.5)
        
    
