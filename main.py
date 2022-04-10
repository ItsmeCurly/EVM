from evm.converters import metadata2str
from evm.magnify import find_heart_rate, magnify_color
from evm.video import Video, save_video

if __name__ == "__main__":
    metadata = {
        "vid_name": "face.mp4",
        "freq_min": 1.0,
        "freq_max": 1.8,
        "pyramid_levels": 3,
        "amplification": 100,
    }
    video = Video(metadata["vid_name"])
    amplified_video, fft, frequencies = magnify_color(video, **metadata)
    heart_rate = find_heart_rate(fft=fft, freqs=frequencies, **metadata)

    save_video(amplified_video, metadata2str(metadata))

    print(f"Calculated heart rate: {heart_rate}")
    # magnify_motion("baby.mp4", 0.4, 3)
