import RPi.GPIO as GPIO
import time

# Define GPIO pin for speaker
SPEAKER_PIN = 21

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SPEAKER_PIN, GPIO.OUT)

# Function to generate a tone
def generate_tone(frequency, duration, volume):
    pwm = GPIO.PWM(SPEAKER_PIN, frequency)
    pwm.start(volume)
    time.sleep(duration)
    pwm.stop()

# Test different frequencies and volumes
frequencies = [800]  # Hz
duty_cycle_percentages = [50]  # Duty cycle percentage

for frequency in frequencies:
    for duty_cycle_percentage in duty_cycle_percentages:
        print(f"Testing frequency: {frequency} Hz, volume: {duty_cycle_percentage}%")
        generate_tone(frequency, 4, duty_cycle_percentage)

# Cleanup GPIO
GPIO.cleanup()