import sys
from os import PathLike
from typing import Optional

import cv2
import numpy as np
import scipy.fftpack as fftpack
from memory_profiler import profile
from scipy import signal


class Video:
    def __init__(
        self,
        path: PathLike,
    ) -> None:
        self.cap = cv2.VideoCapture(path)

        self._load_video_tensor()

    @property
    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def channels(self):
        # return int(self.cap.get(cv2.CAP_PROP_CHANNEL))
        return 3  # RGB

    @property
    def tensor(self):
        return self._tensor

    # build laplacian pyramid for video
    def laplacian(self, levels=3):
        tensor_list = []
        for i in range(0, self._tensor.shape[0]):
            frame = self._tensor[i]
            pyr = self._build_laplacian_pyramid(frame, levels=levels)
            if i == 0:
                for k in range(levels):
                    tensor_list.append(
                        np.zeros(
                            (self._tensor.shape[0], pyr[k].shape[0], pyr[k].shape[1], 3)
                        )
                    )
            for n in range(levels):
                tensor_list[n][i] = pyr[n]
        return tensor_list

    def gaussian(self, levels: Optional[int] = 3):
        for i in range(0, self._tensor.shape[0]):
            frame = self._tensor[i]
            pyr = self._build_gaussian_pyramid(frame, level=levels)
            gaussian_frame = pyr[-1]
            if i == 0:
                vid_data = np.zeros(
                    (
                        self._tensor.shape[0],
                        gaussian_frame.shape[0],
                        gaussian_frame.shape[1],
                        3,
                    )
                )
            vid_data[i] = gaussian_frame
        return vid_data

    def _load_video_tensor(self):
        if not self.cap.isOpened():
            return None

        self._tensor = np.zeros(
            (self.frame_count, self.height, self.width, self.channels), dtype="float"
        )
        for i in range(self.frame_count):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break
            else:
                self._tensor[i] = frame

    # Build Gaussian Pyramid
    def _build_gaussian_pyramid(self, frame: int, level: int = 3):
        s = frame.copy()
        pyramid = [s]
        for i in range(level):
            s = cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    # Build Laplacian Pyramid
    def _build_laplacian_pyramid(self, frame, levels):
        gaussian_pyramid = self._build_gaussian_pyramid(frame, levels)
        pyramid = []
        for i in range(levels, 0, -1):
            GE = cv2.pyrUp(gaussian_pyramid[i])
            L = cv2.subtract(gaussian_pyramid[i - 1], GE)
            pyramid.append(L)
        return pyramid


def amplify_video(video, amplification: int):
    return video * amplification


# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video, freq_min: float, freq_max: float, fps: float, axis: int = 0):
    fft = fftpack.fft(video, axis=0)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    result = np.abs(iff)

    return result, fft, frequencies
