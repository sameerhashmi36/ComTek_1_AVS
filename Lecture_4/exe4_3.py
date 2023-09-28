import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the first recording
duration = 10
sample_rate_1 = 44100
bit_depth_1 = 16

# Parameters for the second recording (different settings)
sample_rate_2 = 22050
bit_depth_2 = 8

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open a stream for the first audio recording
stream_1 = audio.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=sample_rate_1,
                      input=True,
                      frames_per_buffer=1024)

print("Recording (First)...")

# Record the first audio signal
frames_1 = []
for _ in range(0, int(sample_rate_1 / 1024 * duration)):
    data = stream_1.read(1024)
    frames_1.append(data)

print("First recording finished.")

# Close the first audio stream
stream_1.stop_stream()
stream_1.close()

# Open a stream for the second audio recording (different settings)
stream_2 = audio.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=sample_rate_2,
                      input=True,
                      frames_per_buffer=1024)

print("Recording (Second)...")

# Record the second audio signal
frames_2 = []
for _ in range(0, int(sample_rate_2 / 1024 * duration)):
    data = stream_2.read(1024)
    frames_2.append(data)

print("Second recording finished.")

# Close the second audio stream
stream_2.stop_stream()
stream_2.close()

audio.terminate()

# Convert audio frames to numpy arrays for both recordings
audio_data_1 = np.frombuffer(b''.join(frames_1), dtype=np.int16)
audio_data_2 = np.frombuffer(b''.join(frames_2), dtype=np.int16)

# Calculate the time axis for both recordings
time_1 = np.linspace(0, duration, len(audio_data_1))
time_2 = np.linspace(0, duration, len(audio_data_2))

# Calculate the frequency axis for both recordings
n_1 = len(audio_data_1)
n_2 = len(audio_data_2)
freq_1 = np.fft.rfftfreq(n_1, d=1.0/sample_rate_1)
freq_2 = np.fft.rfftfreq(n_2, d=1.0/sample_rate_2)

# Perform FFT to get the frequency domain representations for both recordings
fft_result_1 = np.fft.rfft(audio_data_1)
fft_result_2 = np.fft.rfft(audio_data_2)

# Calculate the amplitude and phase for both recordings
amplitude_1 = np.abs(fft_result_1)
amplitude_2 = np.abs(fft_result_2)
phase_1 = np.angle(fft_result_1)
phase_2 = np.angle(fft_result_2)

# Plot the amplitude and phase for both recordings
plt.figure(figsize=(12, 6))

# Amplitude vs. Frequency
plt.subplot(2, 2, 1)
plt.semilogx(freq_1, amplitude_1, color='r')
plt.title("Amplitude vs. Frequency (Recording 1)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.subplot(2, 2, 2)
plt.semilogx(freq_2, amplitude_2, color='b')
plt.title("Amplitude vs. Frequency (Recording 2)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Phase Response vs. Frequency
plt.subplot(2, 2, 3)
plt.semilogx(freq_1, phase_1, color='g')
plt.title("Phase Response vs. Frequency (Recording 1)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.subplot(2, 2, 4)
plt.semilogx(freq_2, phase_2, color='y')
plt.title("Phase Response vs. Frequency (Recording 2)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

# Show the plots
plt.tight_layout()
plt.show()

"""
    Sample Rate (Sampling Frequency):

    Recording 1 (sample_rate_1): 44,100 Hz (44.1 kHz)
    Recording 2 (sample_rate_2): 22,050 Hz (22.05 kHz)
    Bit Depth:

    Recording 1 (bit_depth_1): 16 bits
    Recording 2 (bit_depth_2): 8 bits
    Now, let's discuss the differences observed in the plots:

    Amplitude vs. Frequency:

    In the "Amplitude vs. Frequency" plots, you can see the differences in the amplitude spectra of the two recordings.
    Recording 1 (Red Curve) at 44.1 kHz sampling rate has a wider frequency range and higher resolution compared to Recording 2 (Blue Curve) at 22.05 kHz sampling rate.
    Recording 1 captures more fine-grained details in the frequency domain due to its higher sampling rate and bit depth.
    Recording 2 has a more limited frequency range and may lose some high-frequency information.
    Phase Response vs. Frequency:

    In the "Phase Response vs. Frequency" plots, you can see the differences in the phase response of the two recordings.
    Both recordings exhibit phase shifts as a function of frequency, but the exact phase values differ.
    The phase response can be important for certain signal processing applications, such as filtering.

"""