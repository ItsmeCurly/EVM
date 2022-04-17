from memory_profiler import profile
from scipy import signal

from evm.filter import butter_bandpass_filter, fft_filter
from evm.pyramid import gaussian_video
from evm.video import Video, amplify, reconstruct_from_tensorlist, reconstruct_video


def magnify_color(
    video: Video,
    freq_min,
    freq_max,
    amplification,
    pyramid_levels=4,
    **kwargs,
):

    gau_video = gaussian_video(video.tensor, pyramid_levels=pyramid_levels)
    filtered_tensor, fft, frequencies = fft_filter(
        gau_video, freq_min, freq_max, video.fps
    )
    amplified_video = amplify(filtered_tensor, amplification=amplification)
    return (
        reconstruct_video(amplified_video, video.tensor, pyramid_levels=pyramid_levels),
        fft,
        frequencies,
    )


# magnify motion
@profile
def magnify_motion(
    video: Video,
    freq_min=0.4,
    freq_max=3.0,
    amplification=20,
    pyramid_levels=4,
):
    return video.tensor + reconstruct_from_tensorlist(
        [
            butter_bandpass_filter(
                video.laplacian(video.tensor, pyramid_levels=pyramid_levels)[i],
                freq_min,
                freq_max,
                video.fps,
            )
            * amplification
            for i in range(pyramid_levels)
        ],
        pyramid_levels=pyramid_levels,
    )


# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max, **kwargs):
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
