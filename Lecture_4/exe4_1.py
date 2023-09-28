import pyaudio
import numpy as np
import wave

# Parameters
duration = 10
sample_rate = 44100
bit_depth = 16

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open a stream for audio recording
stream = audio.open(format=pyaudio.paInt16,
                    channels=1,  # Mono audio
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

print("Recording...")

# Record audio
frames = []
for _ in range(0, int(sample_rate / 1024 * duration)):
    data = stream.read(1024)
    frames.append(data)

print("Recording finished.")

# Close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Convert audio frames to numpy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# Calculate sampling frequency and amplitude resolution
sampling_frequency = sample_rate
amplitude_resolution = 2 ** bit_depth

print(f"Sampling Frequency: {sampling_frequency} Hz")
print(f"Amplitude Resolution: {amplitude_resolution} levels")

with wave.open("recorded_audio.wav", "wb") as wf:
    wf.setnchannels(1)  # Mono audio
    wf.setsampwidth(audio_data.dtype.itemsize)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data.tobytes())

