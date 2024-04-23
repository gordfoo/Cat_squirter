import RPi.GPIO as GPIO
from time import sleep
import cv2
import datetime
import RPi.GPIO as GPIO
import time
import datetime
import threading
import queue

frame_queue = queue.Queue()

thres = 0.5 # Threshold to detect object
nms = 0.2

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

# Load names of classes and get only the class "cat"
classes = []
with open("coco.names", "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load the neural network
net = cv2.dnn_DetectionModel('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

image_list = []
process_image_list = []

def trigger_camera_and_detect():
    # Initialize the video capture object with the first camera device
      
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    process_duration = 10
    
    end_time = time.time() + process_duration
    
    while time.time() < end_time:
        # Capture frame-by-frame
        ret, frame = cap.read()
        print(f'Return is {ret}')
        frame_size = frame.size
        print(f'Frame size is {frame_size}')
        
        
        if not ret:
            print("Error: Failed to capture image")
            return
        
        if not frame_queue.full():
            print('Frame put in queue')
            frame_queue.put(frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    ## Display the resulting frame
    #cv2.imshow('Camera Test', frame)
    
    #How many times to perform the image routine
    

def process_image():
    
    process_duration = 10
    
    end_time = time.time() + process_duration
    
    print('Process image called')
    
    while time.time() < end_time:
        if not frame_queue.empty():
            frame = frame_queue.get()
            print('Got a frame')
            img, objectInfo = getObjects(frame, objects=['cat','person'])
            print('Ran getObjects')
            print(f'objectInfo {objectInfo}')
        
            for _box, className in objectInfo:
                print('Object (className) is {className}')
                if len(className) > 0:
                    process_image_list.append(className)
                
                if className == 'cat':
                    activate_external_devices()
                    break
        

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
            
    

def thread_camera_trigger_and_image_process():
    camera_thread = threading.Thread(target=trigger_camera_and_detect)
    process_thread = threading.Thread(target=process_image)

    camera_thread.start()
    process_thread.start()

    camera_thread.join()
    process_thread.join()

    

def getObjects(img, thres=0.5, nms=0.2, draw=True, objects=['person','cat']):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            print(classId)
            className = classNames[classId - 1]
            print(className)            
            if className in objects:
                activate_external_devices()
                objectInfo.append([box,className])
                image_list.append(className)
                print(f'Class name is {className}')
#                 if (draw):
#                     cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#                     cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#                     cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#                     cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    

    return img,objectInfo
    
def take_face_shot():
    #Smile for the camera, kitty!
    cap2 = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    print('Face shot routine activated')
    
    today = datetime.date.today()
    
    num_pics = 1
    
    while num_pics > 0:
        #Read the image from the video feed
        _result, img = cap2.read()
        
        #Write the image to the file
        filename = f'funny_cat_pic_{num_pics}_{today}.png'
        
        cv2.imwrite(filename, img)
        
    cap2.release()
    
    
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



def activate_external_devices():
    # Create threads for each function
    thread_squirter = threading.Thread(target=trigger_squirter)
    thread_speaker = threading.Thread(target=trigger_speaker)
    thread_camera = threading.Thread(target=take_face_shot)

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

    process_duration = 20
    
    end_time = time.time() + process_duration

    while True:
#     while time.time() < end_time:
        
        if GPIO.input(24):
            #test_camera()
            thread_camera_trigger_and_image_process()
        current_time = time.time()
        counter =+ 1
        if counter % 20 == 0:
            print(f'Check occurred at {current_time}')
            counter = 0
        sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    print(image_list)
        
    
            

        
    


