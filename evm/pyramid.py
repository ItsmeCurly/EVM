# build laplacian pyramid for video
import cv2
import numpy as np


# Build Gaussian Pyramid
def build_gaussian_pyramid(src, pyramid_levels):
    s = src.copy()
    pyramid = [s]
    for i in range(pyramid_levels):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


# build gaussian pyramid for video
def gaussian_video(video_tensor, pyramid_levels):
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_gaussian_pyramid(frame, pyramid_levels=pyramid_levels)
        gaussian_frame = pyr[-1]
        if i == 0:
            vid_data = np.zeros(
                (
                    video_tensor.shape[0],
                    gaussian_frame.shape[0],
                    gaussian_frame.shape[1],
                    3,
                )
            )
        vid_data[i] = gaussian_frame
    return vid_data


# Build Laplacian Pyramid
def build_laplacian_pyramid(src, pyramid_levels):
    gaussianPyramid = build_gaussian_pyramid(src, pyramid_levels)
    pyramid = []
    for i in range(pyramid_levels, 0, -1):
        GE = cv2.pyrUp(gaussianPyramid[i])
        L = cv2.subtract(gaussianPyramid[i - 1], GE)
        pyramid.append(L)
    return pyramid


def laplacian_video(video_tensor, pyramid_levels):
    tensor_list = []
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_laplacian_pyramid(frame, pyramid_levels=pyramid_levels)
        if i == 0:
            for k in range(pyramid_levels):
                tensor_list.append(
                    np.zeros(
                        (video_tensor.shape[0], pyr[k].shape[0], pyr[k].shape[1], 3)
                    )
                )
        for n in range(pyramid_levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list
