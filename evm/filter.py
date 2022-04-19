import cv2
import numpy as np
import scipy.fftpack as fftpack
from matplotlib import pyplot as plt
from scipy import signal

from evm.video import save_video


# butterworth bandpass filter
def butter_bandpass_filter(data, freq_min, freq_max, fs, order=5):
    omega = 0.5 * fs
    freq_min = freq_min / omega
    freq_max = freq_max / omega
    b, a = signal.butter(order, [freq_min, freq_max], btype="band")
    y = signal.lfilter(b, a, data, axis=0)
    return y


# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video, freq_min: float, freq_max: float, fps: float, axis: int = 0):
    fft = fftpack.fft(video, axis=axis)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
    bound_freq_min = (np.abs(frequencies - freq_min)).argmin()
    bound_freq_max = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_freq_min] = 0
    fft[bound_freq_max:-bound_freq_max] = 0
    fft[-bound_freq_min:] = 0
    ifft = fftpack.ifft(fft, axis=0)
    result = np.abs(ifft)

    # res = []
    # for i in result:
    #     res.append(np.average(i))

    # plt.plot(res)
    # plt.show()

    return result, fft, frequencies
