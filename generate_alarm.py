"""
Generate a simple alarm sound (alarm.wav)
Run this once to create the alarm sound file
"""
import numpy as np
import wave
import os

def generate_alarm_sound(filename='static/sounds/alarm.wav', duration=2.0, frequency=1000):
    """Generate a simple sine wave alarm sound"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Parameters
    sample_rate = 44100  # Hz
    num_samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, num_samples, False)
    
    # Create alternating frequency for alarm effect
    alarm = np.zeros(num_samples)
    for i in range(4):  # 4 beeps
        start = int(i * num_samples / 4)
        end = int(start + num_samples / 8)
        alarm[start:end] = np.sin(frequency * 2 * np.pi * t[start:end])
    
    # Normalize to 16-bit range
    alarm = np.int16(alarm * 32767)
    
    # Write to WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(alarm.tobytes())
    
    print(f"[SUCCESS] Alarm sound created: {filename}")

if __name__ == '__main__':
    generate_alarm_sound()
