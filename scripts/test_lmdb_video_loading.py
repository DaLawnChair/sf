


# /home/ma-user/work/dataset/self_forcing/self_forcing/lmdb_temp_video_dataset



"""
This is a clone of the create_lmdb_14b_shards.py file, but for videos.

python create_lmdb_14b_shards_video.py --data_path /home/ma-user/work/dataset/self_forcing/self_forcing/temp_video_dataset --lmdb_path /home/ma-user/work/dataset/self_forcing/self_forcing/lmdb_temp_video_dataset
"""

import sys
import os
project_root = os.environ["TASK_RUN_REPO"]
sys.path.insert(0, project_root)

from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os


from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_video_lmdb, process_data_dict_videos, store_video_arrays_to_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os

from utils.dataset import VideoRegressionShardingLMDBDataset

# from utils.lmdb import store_arrays_to_lmdb, process_data_dict, process_data_dict_videos, store_video_arrays_to_lmdb


dataset = VideoRegressionShardingLMDBDataset("/home/ma-user/work/dataset/self_forcing/self_forcing/lmdb_temp_video_dataset")

import ipdb;ipdb.set_trace()
# def main():
#     """
#     Aggregate all ode pairs inside a folder into a lmdb dataset.
#     Each pt file should contain a (key, value) pair representing a
#     video's ODE trajectories.
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", type=str,
#                         required=True, help="path to video pairs")
#     parser.add_argument("--lmdb_path", type=str,
#                         required=True, help="path to lmdb")
#     parser.add_argument("--num_shards", type=int,
#                         default=16, help="num_shards")


# if __name__ == "__main__":
#     main()
