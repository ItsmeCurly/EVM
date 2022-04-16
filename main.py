from inspect import currentframe
from evm.converters import metadata2str
from evm.magnify import find_heart_rate, magnify_color
from evm.video import Video, save_video
import cv2
from time import sleep

def on_change_freq_min(value):
    metadata["freq_min"] = value / 100

def on_change_freq_max(value):
    metadata["freq_max"] = value / 100

def on_change_amp(value):
    metadata["amplification"] = value

def do_task():
    amplified_video, fft, frequencies = magnify_color(video, **metadata)
    heart_rate = find_heart_rate(fft=fft, freqs=frequencies, **metadata)
    return amplified_video, heart_rate

if __name__ == "__main__":
    metadata = {
        "vid_name": "15_meter_output",
        "freq_min": 12/12,
        "freq_max": 13/12,
        "pyramid_levels": 3,
        "amplification": 50,
    }
    old_metadata = metadata.copy()
    video = Video(metadata["vid_name"] + ".mp4")

    amplified_video, heart_rate = do_task()

    cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test', 600,600)
    cv2.createTrackbar('freq_min', "test", 0, 200, on_change_freq_min)
    cv2.createTrackbar('freq_max', "test", 0, 200, on_change_freq_max)
    cv2.createTrackbar('amp', "test", 0, 100, on_change_amp)

    currentframe = 0
    flag = True
    while(flag):
        amplified_video, heart_rate = do_task()
        for i in range(len(amplified_video)):
            cv2.imshow("test", cv2.convertScaleAbs(amplified_video[currentframe]))
            currentframe += 1
            if currentframe > len(amplified_video) - 1:
                currentframe = 0
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                flag = False
                break
            if old_metadata != metadata:
                old_metadata = metadata.copy()
                break

    save_video(amplified_video, metadata2str(metadata))
        
    print(f"Calculated heart rate: {heart_rate}")
    # magnify_motion("baby.mp4", 0.4, 3)
