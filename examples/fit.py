import os
import argparse

import pandas as pd

def argsparser():
    parser = argparse.ArgumentParser("Fitting a model for video recognition")
    parser.add_argument('--alg', type=str, default='tsn',
            choices=['tsn', 'tsm', 'i3d', 'eco', 'eco_full', 'c3d', 'r2p1d', 'r3d', 'stgcn'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='')
    parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/hmdb6/')
    parser.add_argument('--log_path', help='The path of saving logs', type=str, default='log.txt')
    parser.add_argument('--save_path', help='The path for saving the trained pipeline', type=str, default='fitted_pipeline')

    return parser

def run(args):
    # Set the logger path
    from autovideo.utils import set_log_path, logger
    set_log_path(args.log_path)

    train_table_path = os.path.join(args.data_dir, 'train.csv')
    train_media_dir = os.path.join(args.data_dir, 'media')
    target_index = 2

    from autovideo import fit, build_pipeline, compute_accuracy_with_preds
    # Read the CSV file
    train_dataset = pd.read_csv(train_table_path)

    # Build pipeline based on configs
    # Here we can specify the hyperparameters defined in each primitive
    # The default hyperparameters will be used if not specified
    config = {
        "transformation":[
            ("RandomCrop", {"size": (128,128)}),
            ("Scale", {"size": (128,128)}),
        ],
        "augmentation": [
            ("meta_ChannelShuffle", {"p": 0.5} ),
            ("blur_GaussianBlur",),
            ("flip_Fliplr", ),
            ("imgcorruptlike_GaussianNoise", ),
        ],
        "multi_aug": "meta_Sometimes",
        "algorithm": args.alg,
        "load_pretrained": args.pretrained,
        "epochs": args.epochs,
    }
    pipeline = build_pipeline(config)

    # Fit
    _, fitted_pipeline = fit(train_dataset=train_dataset,
                             train_media_dir=train_media_dir,
                             target_index=target_index,
                             pipeline=pipeline)

    # Save the fitted pipeline
    import torch
    torch.save(fitted_pipeline, args.save_path)

if __name__ == '__main__':
    parser = argsparser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Fit and produce
    run(args)

