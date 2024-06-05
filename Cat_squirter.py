import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time
from picamera import PiCamera
import threading

# Setup GPIO pins
GPIO.setmode(GPIO.BCM)
IR_SENSOR_PIN = 24
SPEAKER_PIN = 21
RELAY_PIN = 14

GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
GPIO.setup(SPEAKER_PIN, GPIO.OUT)
GPIO.setup(RELAY_PIN, GPIO.OUT)

# Load TensorFlow Lite model
model_path = 'MobileNet_V2.tflite'  # Update this path if necessary
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup camera
camera = PiCamera()
camera.resolution = (224, 224)

# Load class labels
with open('imagenet_labels.txt', 'r') as f:
    #class_labels = [line.strip().split('\t')[1] for line in f.readlines()]
    class_labels = [line.split()[1] for line in f.readlines()]

def capture_image():
    image = np.empty((224, 224, 3), dtype=np.uint8)
    camera.capture(image, 'rgb')
    return image

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    if input_details[0]['dtype'] == np.uint8:
        image = image.astype(np.uint8)  # Ensure the image is of type UINT8 if required
    else:
        image = image.astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1] if needed
    return image

def is_cat(predictions, threshold=0.7):
    cat_labels = ['Egyptian_cat', 'Egyptian', 'Persian', 'cat'] 
    for prediction in predictions:
        print(f'Prediction {prediction[0]}')
        #if prediction[1] == 'Egyptian_cat' and prediction[2] > threshold:
        if prediction[1] in cat_labels and prediction[2] > threshold:
            print(f'{prediction[2]}')
            print('CATTTTTTTTTTTTTTT \n CATTTTTTTTT')
            
            return True
    return False
    
def activate_speaker():
    
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
    
    #GPIO.output(SPEAKER_PIN, GPIO.HIGH)
    #time.sleep(0.5)
    #GPIO.output(SPEAKER_PIN, GPIO.LOW)
    #time.sleep(0.5)

def activate_solenoid():
    time.sleep()  # Wait for 1 second after beeping starts
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(1.5)
    GPIO.output(RELAY_PIN, GPIO.LOW)
    
# Function to decode predictions
def decode_predictions(predictions, top=5):
    top_indices = predictions.argsort()[-top:][::-1]
    result = [(i, class_labels[i], predictions[i]) for i in top_indices]
    return result


def thread_solenoid_and_speaker_process():
    solenoid_thread = threading.Thread(target=activate_solenoid)
    speaker_thread = threading.Thread(target=activate_speaker)
    camera_thread = threading.Thread(target=save_image_process)
    
    solenoid_thread.start()
    speaker_thread.start()
    camera_thread.start()
    
    solenoid_thread.join()
    speaker_thread.join()
    camera_thread.join()

def save_image_process():
    image = np.empty((224, 224, 3), dtype=np.uint8)
    camera.capture(image, 'rgb')
    #return image

def main():
    while True:
        if GPIO.input(IR_SENSOR_PIN):
            start_time = time.time()
            while time.time() - start_time < 10:
                image = capture_image()
                processed_image = preprocess_image(image)
                
                interpreter.set_tensor(input_details[0]['index'], processed_image)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                print(f'Checking Image. Predictions {predictions}')
                
                predictions_text = np.squeeze(predictions)
                top_predictions = decode_predictions(predictions_text)
                print(top_predictions)  # Print the top predictions
                
                if is_cat(top_predictions):
                    
                    thread_solenoid_and_speaker_process()
                
                    time.sleep(1.5)  # Wait additional 0.5 seconds after solenoid activation
                    break  # Exit the loop if cat is detected
                
                #if is_cat(predictions):
                #   for _ in range(5):  # Beep for 5 seconds
                #        activate_speaker()
                #    
                #    time.sleep(1)  # Wait for 1 second after beeping starts
                #    activate_solenoid()
                #    time.sleep(1.5 - 1)  # Wait additional 0.5 seconds after solenoid activation
                #    break  # Exit the loop if cat is detected
            print('No cat detected')
            time.sleep(1)  # To prevent excessive CPU usage when no detection
        else:
            time.sleep(0.1)  # To prevent excessive CPU usage when no IR signal

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        GPIO.cleanup()
        camera.close()
	