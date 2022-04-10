import sys
from os import PathLike
from typing import Optional

import cv2
import numpy as np
import scipy.fftpack as fftpack
from memory_profiler import profile
from scipy import signal

from evm.utils import find_file
from evm.video import Video, amplify_video, fft_filter


# convert RBG to YIQ
def rgb2ntsc(src):
    [rows, cols] = src.shape[:2]
    dst = np.zeros((rows, cols, 3), dtype=np.float64)
    T = np.array(
        [[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]]
    )
    for i in range(rows):
        for j in range(cols):
            dst[i, j] = np.dot(T, src[i, j])
    return dst


# convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst = np.zeros((rows, cols, 3), dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j] = np.dot(T, src[i, j])
    return dst


# save video to files
def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out.mp4", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()


def magnify_color(video_name, low, high, levels=3, amplification=50):
    if isinstance(video_name, str):
        video_path = find_file(video_name)
        if video_path:
            video_name = str(video_path[0].resolve())
    video = Video(video_name)
    gaussian_video = video.gaussian(levels=levels)
    filtered_tensor, fft, frequencies = fft_filter(gaussian_video, low, high, video.fps)
    heart_rate = find_heart_rate(fft, frequencies, 1, 1.8)

    amplified_video = amplify_video(filtered_tensor, amplification=amplification)
    save_video(reconstruct_video(amplified_video, video.tensor, levels=levels))


# butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.lfilter(b, a, data, axis=0)
    return y


# reconstract video from laplacian pyramid
def reconstruct_from_tensorlist(filter_tensor_list, levels=3):
    final = np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels - 1):
            up = (
                cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
            )  # can be changed to up=cv2.pyrUp(up)
        final[i] = up
    return final


# reconstract video from original video and gaussian video
def reconstruct_video(amp_video, origin_video, levels=3):
    final_video = np.zeros(origin_video.shape)
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img = cv2.pyrUp(img)
        img = img + origin_video[i]
        final_video[i] = img
    return final_video


# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    return freqs[max_peak] * 60


# manify motion
@profile
def magnify_motion(video_name, low, high, levels=1, amplification=20):
    video = Video(video_name)

    save_video(
        video.tensor
        + reconstruct_from_tensorlist(
            [
                butter_bandpass_filter(
                    video.laplacian(video.tensor, levels=levels)[i],
                    low,
                    high,
                    video.fps,
                )
                * amplification
                for i in range(levels)
            ],
            levels=levels,
        )
    )


if __name__ == "__main__":
    magnify_color("baby.mp4", 0.4, 3)
    # magnify_motion("baby.mp4", 0.4, 3)
