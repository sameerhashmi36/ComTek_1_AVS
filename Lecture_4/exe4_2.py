import wave
import numpy as np
import matplotlib.pyplot as plt

# Load the audio from the WAV file
with wave.open("recorded_audio.wav", "rb") as wf:
    audio_params = wf.getparams()
    audio_data = wf.readframes(audio_params.nframes)
    audio_data = np.frombuffer(audio_data, dtype=np.int16)
    sample_rate = audio_params.framerate

# Calculate the time axis
duration = len(audio_data) / sample_rate
time = np.linspace(0, duration, len(audio_data))

# Calculate the frequency axis
n = len(audio_data)
freq = np.fft.rfftfreq(n, d=1.0/sample_rate)

# Perform FFT to get the frequency domain representation
fft_result = np.fft.rfft(audio_data)
print("fft_result: ", fft_result)

# Calculate the amplitude and phase
amplitude = np.abs(fft_result)
phase = np.angle(fft_result)

# Printing phase, amplitude, freq
print("amplitude: ", amplitude)
print("freq: ", freq)
print("phase: ", phase)

# Plot the amplitude as a function of time
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, audio_data, color='b')
plt.title("Amplitude vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the amplitude as a function of frequency
plt.subplot(2, 1, 2)
plt.semilogx(freq, amplitude, color='r')
plt.title("Amplitude vs. Frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

# Show the plots
plt.tight_layout()
plt.show()

# Plot the phase response as a function of frequency
plt.figure(figsize=(8, 4))
plt.semilogx(freq, phase, color='g')
plt.title("Phase Response vs. Frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.show()
