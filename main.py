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
        name: Optional[str] = None,
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

    # apply temporal ideal bandpass filter to gaussian video
    def temporal_ideal_filter(self, low, high, fps, axis=0):
        gaussian_tensor = self.gaussian()
        fft = fftpack.fft(gaussian_tensor, axis=axis)
        frequencies = fftpack.fftfreq(gaussian_tensor.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff = fftpack.ifft(fft, axis=axis)
        return np.abs(iff)

    # amplify the video
    def amplified_gaussian(
        self, levels: Optional[int] = 3, amplification: Optional[int] = 50
    ):
        return self.gaussian(levels=levels) * amplification

    # reconstract video from original video and gaussian video
    def reconstruct_video(self, levels=3):
        final_video = np.zeros(self._tensor.shape)

        amp_video = self.amplified_gaussian(amplification=50)
        for i in range(0, amp_video.shape[0]):
            img = amp_video[i]
            for x in range(levels):
                img = cv2.pyrUp(img)
            img = img + self._tensor[i]
            final_video[i] = img
        return final_video

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
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out.mp4", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()


# magnify color
# def magnify_color(video_name, low, high, levels=3, amplification=20):
#     t, f = load_video(video_name)
#     gau_video = gaussian_video(t, levels=levels)
#     filtered_tensor = temporal_ideal_filter(gau_video, low, high, f)
#     amplified_video = amplify_video(filtered_tensor, amplification=amplification)
#     final = reconstruct_video(amplified_video, t, levels=3)
#     save_video(final)


# butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.lfilter(b, a, data, axis=0)
    return y


# reconstract video from laplacian pyramid
def reconstract_from_tensorlist(filter_tensor_list, levels=3):
    final = np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels - 1):
            up = (
                cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
            )  # can be changed to up=cv2.pyrUp(up)
        final[i] = up
    return final


# manify motion
@profile
def magnify_motion(video_name, low, high, levels=1, amplification=20):
    video = Video(video_name)

    save_video(
        video.tensor
        + reconstract_from_tensorlist(
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
    # magnify_color("baby.mp4",0.4,3)
    magnify_motion("baby.mp4", 0.4, 3)
