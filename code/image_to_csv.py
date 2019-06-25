import pandas as pd
import numpy as np
import sys
import os
import conf

INPUT = conf.source_image
OUTPUT = conf.source_csv


def make_csv():
    data_np = []
    for sub_folder in os.listdir(INPUT):
        for img in os.listdir(os.path.join(INPUT, sub_folder)):
            data_np.append([sub_folder+"/"+img, sub_folder])
    data_np = np.array(data_np)
    return data_np


def save_csv(data):
    data_pd = pd.DataFrame(data, columns=['path', "label"])
    data_pd.to_csv(OUTPUT, index=False)


if __name__ == '__main__':
    sys.stderr.write('make csv file')
    data = make_csv()
    save_csv(data)
