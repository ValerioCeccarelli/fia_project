import zipfile
from urllib import request
import os
import sys

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
DATASET_PATH = 'OnlineNewsPopularity.zip'


def download_dataset():
    request.urlretrieve(DATASET_URL, DATASET_PATH)


def extract_dataset(target_dir):
    with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def clear():
    os.remove(DATASET_PATH)


def download_and_extract_dataset(target_dir):
    download_dataset()
    extract_dataset(target_dir)
    clear()


if __name__ == '__main__':
    target_dir = '.'
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    download_and_extract_dataset(target_dir)
