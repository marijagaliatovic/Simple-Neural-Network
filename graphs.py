import os #za operacije patha
import torchaudio
import torch
import pandas as pd #za tablicne podatke
import matplotlib.pyplot as plt

ANNOTATIONS_FILE = "C:/Users/MarijaGaliatović/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "C:/Users/MarijaGaliatović/Downloads/UrbanSound8K/audio"

def _resample_if_necessary(signal, sr, target_sample_rate):
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

sample_rate=22050
annotations = pd.read_csv(ANNOTATIONS_FILE)
fold = f"fold{annotations.iloc[0, 5]}"
audio_sample_path = os.path.join(AUDIO_DIR, fold, annotations.iloc[0,0])
label = annotations.iloc[0,6]
signal, sr = torchaudio.load(audio_sample_path)

# Create the time axis
print(f"signal.size(1):{signal.size(1)}, sr = {sr}")
time_axis = torch.arange(0, signal.size(1))

# Plot the audio signal
plt.figure(figsize=(10, 6))
plt.plot(time_axis, signal.t().numpy())
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title(f"Audio Signal - Label: {label}")
plt.show()

signal = _resample_if_necessary(signal, sr, sample_rate)

print(f"signal.size(1):{signal.size(1)}, sr = {sr}")
time_axis = torch.arange(0, signal.size(1))

# Plot the audio signal
plt.figure(figsize=(10, 6))
plt.plot(time_axis, signal.t().numpy())
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title(f"Audio Signal - Label: {label}")
plt.show()