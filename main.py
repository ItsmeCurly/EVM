from inspect import currentframe
from time import sleep

import cv2
from matplotlib import pyplot as plt

from evm.converters import metadata2str
from evm.magnify import find_heart_rate, find_heart_rate_2, magnify_color, magnify_motion
from evm.video import Video, save_video


def on_change_freq_min(value):
    metadata["freq_min"] = value / 60
    #metadata["freq_min"] = (value - 5) / 60
    #metadata["freq_max"] = (value + 5) / 60


def on_change_freq_max(value):
    metadata["freq_max"] = value / 60


def on_change_amp(value):
    metadata["amplification"] = value


def do_task():
    amplified_video, filtered_video, fft, frequencies = magnify_color(video, **metadata)

    heart_rate = find_heart_rate(fft=fft, freqs=frequencies, **metadata)
    heart_rate_2 = find_heart_rate_2(ifft=filtered_video, fps=video.fps, **metadata)

    #print(heart_rate, heart_rate_2)

    return amplified_video, heart_rate


if __name__ == "__main__":
    metadata = {
        "vid_name": "1080p_output",
        "freq_min": 0.8,
        "freq_max": 1,
        "pyramid_levels": 3,
        "amplification": 50,
    }
    old_metadata = metadata.copy()
    video = Video(metadata["vid_name"] + ".mp4")

    """

    amplified_video, heart_rate = do_task()

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 600, 600)
    cv2.createTrackbar(
        "freq_min", "test", int(metadata["freq_min"] * 100), 300, on_change_freq_min
    )
    cv2.createTrackbar(
        "freq_max", "test", int(metadata["freq_max"] * 100), 300, on_change_freq_max
    )
    cv2.createTrackbar(
        "amp", "test", int(metadata["amplification"]), 150, on_change_amp
    )

    currentframe = 0
    flag = True
    while flag:
        amplified_video, heart_rate = do_task()
        for i in range(len(amplified_video)):
            cv2.imshow("test", cv2.convertScaleAbs(amplified_video[currentframe]))
            currentframe += 1
            if currentframe > len(amplified_video) - 1:
                currentframe = 0
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                flag = False
                break
            if old_metadata != metadata:
                old_metadata = metadata.copy()
                break

    save_video(amplified_video, metadata2str(metadata))

    print(f"Calculated heart rate: {heart_rate}")
    """
    magnify_motion(video, 0.4, 3)
