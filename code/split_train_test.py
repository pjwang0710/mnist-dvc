from shutil import copyfile
import conf
import numpy as np
import os

data_dir = conf.data_dir
source_image = conf.source_image
split_rate = 0.8


def init():
    if not os.path.exists(os.path.join(input, "train")):
        os.mkdir(os.path.join(input, "train"))
    if not os.path.exists(os.path.join(input, "validation")):
        os.mkdir(os.path.join(input, "validation"))


def split():
    for sub_folder in os.listdir(source_image):
        for i, img in enumerate(os.listdir(os.path.join(source_image, sub_folder))):
            if i < int(len(os.listdir(os.path.join(source_image, sub_folder))*split_rate):
                copyfile(os.path.join(source_image, sub_folder, img), os.path.join(data_dir, folder_type, "train", img))
            else:
                copyfile(os.path.join(source_image, sub_folder, img), os.path.join(data_dir, folder_type, "validation", img))

sys.stderr.write('split data')
split()
        