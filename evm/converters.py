from typing import Any

import numpy as np


def metadata2str(metadata: dict[str, Any]):
    def _gen_key_val():
        for k, v in metadata.items():
            yield f"{k}-{v}"

    return "_".join(list(_gen_key_val()))


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
