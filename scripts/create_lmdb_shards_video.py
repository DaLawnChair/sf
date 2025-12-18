"""
This is a clone of the create_lmdb_14b_shards.py file, but for videos.

python create_lmdb_shards_video.py \
--data_path /home/ma-user/work/dataset/self_forcing/self_forcing/temp_video_dataset \
--lmdb_path /home/ma-user/work/dataset/self_forcing/self_forcing/lmdb_temp_video_dataset
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

from utils.lmdb import store_arrays_to_lmdb, process_data_dict, process_data_dict_videos, store_video_arrays_to_lmdb


# Edit these for your own model outputs
EXPECTED_VIDEO_SHAPE = (1, 81, 480, 832, 3)
EXPECTED_NOISE_SHAPE = (1, 21, 16, 60, 104)

def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to video pairs")
    parser.add_argument("--lmdb_path", type=str,
                        required=True, help="path to lmdb")
    parser.add_argument("--num_shards", type=int,
                        default=16, help="num_shards")

    args = parser.parse_args()

    all_dirs = sorted(os.listdir(args.data_path))

    # figure out the maximum map size needed
    map_size = int(1e12)  # adapt to your need, set to 1TB by default
    os.makedirs(args.lmdb_path, exist_ok=True)
    # 1) Open one LMDB env per shard
    envs = []
    num_shards = args.num_shards
    for shard_id in range(num_shards):
        print("shard_id ", shard_id)
        path = os.path.join(args.lmdb_path, f"shard_{shard_id}")
        env = lmdb.open(path,
                        map_size=map_size,
                        subdir=True,       # set to True if you want a directory per env
                        readonly=False,
                        metasync=True,
                        sync=True,
                        lock=True,
                        readahead=False,
                        meminit=False)
        envs.append(env)

    counters = [0] * num_shards
    seen_prompts = set()  # for deduplication
    total_samples = 0
    all_files = []

    # for part_dir in all_dirs:
    #     all_files += sorted(glob.glob(os.path.join(args.data_path, part_dir, "*.pt")))

    # for part_dir in all_dirs:
    all_files = sorted([os.path.join(args.data_path, part_dir) for part_dir in all_dirs])
    
    print("len of all_files:", len(all_files))
    # import ipdb;ipdb.set_trace()
    # 2) Prepare a write transaction for each shard
    for idx, file in tqdm(enumerate(all_files)):
        try:
            data_dict = torch.load(file)
            data_dict = process_data_dict_videos(data_dict, seen_prompts)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
            
        # if 'prompts' in data_dict.keys():
        # print(data_dict.keys())
        
        if data_dict["video"].shape != EXPECTED_VIDEO_SHAPE:
            print(f"error: video shape {data_dict['video'].shape} != {EXPECTED_VIDEO_SHAPE}")
            continue
        if data_dict["noise"].shape != EXPECTED_NOISE_SHAPE:
            print(f"error: noise shape {data_dict['noise'].shape} != {EXPECTED_NOISE_SHAPE}")
            continue
        
        shard_id = idx % num_shards
        # write to lmdb file
        store_video_arrays_to_lmdb(envs[shard_id], data_dict, start_index=counters[shard_id])
        counters[shard_id] += len(data_dict['prompts'])
        
        
        # for simplicity, we assume that each video is of the same video sizing
        data_shapes = {
            "video": data_dict["video"].shape,
            "noise": data_dict["noise"].shape,
            "prompts": data_dict["video"].shape, # ecoding prmpts as video.shape is wrong, but works for now
        }
        

    total_samples += len(all_files)

    print(len(seen_prompts))

    # save each entry's shape to lmdb
    for shard_id, env in enumerate(envs):
        with env.begin(write=True) as txn:
            for key, val in (data_dict.items()):
                assert len(data_shapes[key]) == 5
                
                array_shape = np.array(data_shapes[key])  # val.shape)
                array_shape[0] = counters[shard_id]
                shape_key = f"{key}_shape".encode()
                
                print(key, shape_key, array_shape)
                shape_str = " ".join(map(str, array_shape))
                txn.put(shape_key, shape_str.encode())        

    print(f"Finished writing {total_samples} examples into {num_shards} shards under {args.lmdb_path}")


if __name__ == "__main__":
    main()
