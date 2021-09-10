import os
import argparse

import pandas as pd


def argsparser():
    parser = argparse.ArgumentParser("Producing the predictions with a fitted pipeline")
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='')
    parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/hmdb6/')
    parser.add_argument('--log_path', help='The path of saving logs', type=str, default='log.txt')
    parser.add_argument('--load_path', help='The path for loading the trained pipeline', type=str, default='fitted_pipeline')

    return parser

def run(args):
    # Set the logger path
    from autovideo.utils import set_log_path, logger
    set_log_path(args.log_path)

    test_table_path = os.path.join(args.data_dir, 'test.csv')
    test_media_dir = os.path.join(args.data_dir, 'media')
    target_index = 2

    from autovideo import produce, compute_accuracy_with_preds
    # Read the CSV file
    test_dataset_ = pd.read_csv(test_table_path)
    test_dataset = test_dataset_.drop(['label'], axis=1)
    test_labels = test_dataset_['label']


    # Load fitted pipeline
    import torch
    if torch.cuda.is_available():
        fitted_pipeline = torch.load(args.load_path, map_location="cuda:0")
    else:
        fitted_pipeline = torch.load(args.load_path, map_location="cpu")


    # Produce
    predictions = produce(test_dataset=test_dataset,
                          test_media_dir=test_media_dir,
                          target_index=target_index,
                          fitted_pipeline=fitted_pipeline)

    # Get accuracy
    test_acc = compute_accuracy_with_preds(predictions['label'], test_labels)
    logger.info('Testing accuracy {:5.4f}'.format(test_acc))

if __name__ == '__main__':
    parser = argsparser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Produce
    run(args)

