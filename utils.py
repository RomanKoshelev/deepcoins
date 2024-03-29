import os
import sys
import shutil
import IPython
from IPython.core.display import Javascript

def make_dir(path: str):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return path

def remove_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
