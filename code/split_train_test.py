import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import conf
INPUT = conf.source_csv
train_csv = conf.train_csv
test_csv = conf.test_csv


def train_test_split_df(df, test_ratio, seed):
    train, test = train_test_split(df, test_size=test_ratio, random_state=seed)
    return train, test


if __name__ == '__main__':
    test_ratio = float(sys.argv[1])
    seed = int(sys.argv[2])
    df = pd.read_csv(INPUT)
    train, test = train_test_split_df(df, test_ratio, seed)
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
