import sys
from os import PathLike
from typing import Optional, Union

import cv2
import numpy as np
import scipy.fftpack as fftpack
from memory_profiler import profile
from scipy import signal

from evm.utils import find_file


class Video:
    def __init__(
        self,
        name_or_path: Union[str, PathLike],
    ) -> None:
        if isinstance(name_or_path, str):
            video_path = find_file(name_or_path)
            if video_path:
                name_or_path = str(video_path[0].resolve())

        self.cap = cv2.VideoCapture(name_or_path)

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

    def _load_video_tensor(self):
        if not self.cap.isOpened():
            return None

        self._tensor = np.empty(
            (self.frame_count, self.height, self.width, self.channels),
            dtype=np.dtype("uint8"),
        )
        for i in range(self.frame_count):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break
            else:
                self._tensor[i] = frame


def amplify(video, amplification: int):
    return video * amplification


# reconstract video from laplacian pyramid
def reconstruct_from_tensorlist(filter_tensor_list, pyramid_levels):
    final = np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(pyramid_levels - 1):
            up = (
                cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
            )  # can be changed to up=cv2.pyrUp(up)
        final[i] = up
    return final


# reconstract video from original video and gaussian video
def reconstruct_video(amp_video, origin_video, pyramid_levels):
    final_video = np.zeros(origin_video.shape)
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        for x in range(pyramid_levels):
            img = cv2.pyrUp(img)
        img = img + origin_video[i]
        final_video[i] = img
    return final_video


# save video to files
def save_video(video_tensor, name="out"):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(f"{name}.mp4", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()
