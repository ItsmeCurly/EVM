import os
import pathlib


def find_file(filename: str, path=os.getcwd()):
    return sorted(pathlib.Path(path).glob(f"**/*{filename}"))
