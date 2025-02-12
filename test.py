#!/bin/bash/python

import os
from utils import *
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help='Path of config Json')
    config_path = parser.parse_args().config
    with open(config_path, 'r') as config_fp:
        config = json.load(config_fp)
    import pdb
    pdb.set_trace()