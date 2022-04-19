from evm.converters import metadata2str
from evm.magnify import magnify_motion
from evm.video import Video, save_video

metadata = {
    "vid_name": "1080p_output",
    "freq_min": 1.0,
    "freq_max": 2.0,
    "pyramid_levels": 4,
    "amplification": 10,
}
video = Video(metadata["vid_name"] + ".mp4")

processed = magnify_motion(video, **metadata)

save_video(processed, metadata2str(metadata))
