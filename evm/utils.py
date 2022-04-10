import os
import pathlib

import cv2
from scipy import signal


def find_file(filename: str, path=os.getcwd()):
    return sorted(pathlib.Path(path).glob(f"**/*{filename}"))
