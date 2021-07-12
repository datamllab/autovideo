import os
import argparse

import pandas as pd

def argsparser():
    parser = argparse.ArgumentParser("Fitting a model for video recognition and producing the predictions")
    parser.add_argument('--alg', type=str, default='tsn',
            choices=['tsn', 'tsm', 'i3d', 'eco', 'eco_full', 'c3d', 'r2p1d', 'r3d'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='')
    parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/hmdb6/')
    parser.add_argument('--log_path', help='The path of saving logs', type=str, default='log.txt')

    return parser

def run(args):
    # Set the logger path
    from autovideo.utils import set_log_path, logger
    set_log_path(args.log_path)

    train_table_path = os.path.join(args.data_dir, 'train.csv')
    test_table_path = os.path.join(args.data_dir, 'test.csv')
    train_media_dir = os.path.join(args.data_dir, 'media')
    test_media_dir = train_media_dir
    target_index = 2

    from autovideo import fit_produce, extract_frames, build_pipeline, compute_accuracy_with_preds
    # Read the CSV file
    train_dataset = pd.read_csv(train_table_path)
    test_dataset_ = pd.read_csv(test_table_path)
    test_dataset = test_dataset_.drop(['label'], axis=1)
    test_labels = test_dataset_['label']

    # Extract frames from the video
    video_ext = train_dataset.iloc[0, 1].split('.')[-1]
    extract_frames(train_media_dir, video_ext)
    extract_frames(test_media_dir, video_ext)

    # Build pipeline based on configs
    # Here we can specify the hyperparameters defined in each primitive
    # The default hyperparameters will be used if not specified
    config = {
        "algorithm": args.alg,
        "load_pretrained": args.pretrained,
    }
    pipeline = build_pipeline(config)

    # Fit and produce
    predictions = fit_produce(train_dataset=train_dataset,
                              train_media_dir=train_media_dir,
                              test_dataset=test_dataset,
                              test_media_dir=test_media_dir,
                              target_index=target_index,
                              pipeline=pipeline)

    # Get accuracy
    test_acc = compute_accuracy_with_preds(predictions['label'], test_labels)
    logger.info('Testing accuracy {:5.4f}'.format(test_acc))

if __name__ == '__main__':
    parser = argsparser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Fit and produce
    run(args)

