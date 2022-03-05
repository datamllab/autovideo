import os
import numpy as np
import argparse

import ray
from ray import tune
from hyperopt import hp

import pandas as pd
import argparse


def argsparser():
    parser = argparse.ArgumentParser("Automatically searching hyperparameters for video recognition")
    parser.add_argument('--alg', type=str, default='hyperopt',
            choices=['random', 'hyperopt'])
    parser.add_argument('--num_samples', type=int, default=15)
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='0')
    parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/hmdb6/')

    return parser

def run(args):
    from autovideo.searcher import RaySearcher

    train_table_path = os.path.join(args.data_dir, 'train.csv')
    valid_table_path = os.path.join(args.data_dir, 'test.csv')
    train_media_dir = os.path.join(args.data_dir, 'media')
    valid_media_dir = train_media_dir

    train_dataset = pd.read_csv(train_table_path)
    valid_dataset = pd.read_csv(valid_table_path)

    searcher = RaySearcher(
        train_dataset=train_dataset,
        train_media_dir=train_media_dir,
        valid_dataset=valid_dataset,
        valid_media_dir=valid_media_dir
    )

    #Search Space
    search_space = {
        "augmentation": {
            "aug_0": tune.choice([
                ("arithmetic_AdditiveGaussianNoise",),
                ("arithmetic_AdditiveLaplaceNoise",),
            ]),
            "aug_1": tune.choice([
                ("geometric_Rotate",),
                ("geometric_Jigsaw",),
            ]),
        },
        "multi_aug": tune.choice([
            "meta_Sometimes",
            "meta_Sequential",
        ]),
        "algorithm": tune.choice(["tsn"]),
        "learning_rate": tune.uniform(0.0001, 0.001),
        "momentum": tune.uniform(0.9,0.99),
        "weight_decay": tune.uniform(5e-4,1e-3),
        "num_segments": tune.choice([8,16,32]),
    }

    # Tuning
    config = {
        "searching_algorithm": args.alg,
        "num_samples": args.num_samples,
    }

    best_config = searcher.search(
        search_space=search_space,
        config=config
    )

    print("Best config: ", best_config)
    
if __name__ == '__main__':
    parser = argsparser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Search
    run(args)
