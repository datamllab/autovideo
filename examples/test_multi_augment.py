import os
import argparse

import pandas as pd

def argsparser():
    parser = argparse.ArgumentParser("Fitting a model for video recognition")
    parser.add_argument('--aug', type=str, default='meta_ChannelShuffle')
    parser.add_argument('--alg', type=str, default='tsn',
            choices=['tsn', 'tsm', 'i3d', 'eco', 'eco_full', 'c3d', 'r2p1d', 'r3d', 'stgcn'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='2')
    parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/hmdb6/')
    parser.add_argument('--log_path', help='The path of saving logs', type=str, default='log.txt')
    parser.add_argument('--save_path', help='The path for saving the trained pipeline', type=str, default='fitted_pipeline')

    return parser

def test_augmentation(config, augmentation_methods=["meta_ChannelShuffle", "blur_GaussianBlur", "flip_Fliplr", "imgcorruptlike_GaussianNoise"], multi_aug=None):
    """Build a pipline based on the config
    """
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep
    algorithm = config.pop('algorithm', None)
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    #Step 0: Denormalise
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.denormalize'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    #Step 1: Dataset to DataFrame
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.dataset_to_dataframe'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_0.index}.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    #Step 2: Column Parser
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.column_parser'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_1.index}.produce')
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    #Step 3: Extract columns by semantic types - Attributes
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.extract_columns_by_semantic_types'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_2.index}.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                      data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    pipeline_description.add_step(step_3)

    #Step 4: Extract Columns by semantic types - Target
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.extract_columns_by_semantic_types'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_1.index}.produce')
    step_4.add_output('produce')
    step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                      data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    pipeline_description.add_step(step_4)

    #Step 5: Extract frames by extension / directly load numpy
    if algorithm == 'stgcn':
        step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.numpy_loader'))
    else:
        step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.extract_frames'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_3.index}.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.transformation.Scale'))
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_5.index}.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)

    # Step 6: Video Augmentation
    curr_step_no = int(f'{step_6.index}')
    for i in range(len(augmentation_methods)):
        alg_python_path = 'd3m.primitives.autovideo.augmentation.' + augmentation_methods[i]
        step_augmentation = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
        step_augmentation.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        print(alg_python_path, 'steps.'+str(curr_step_no)+'.produce')
        #for key, value in config.items():
        #    step_6.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_augmentation.add_output('produce')
        pipeline_description.add_step(step_augmentation)
        curr_step_no += 1
    print(pipeline_description.steps)

    # Step 7: Video Augmentation
    alg_python_path = 'd3m.primitives.autovideo.augmentation.' + augmentation_methods[1]
    step_7 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
    step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
    #for key, value in config.items():
    #    step_6.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
    step_7.add_output('produce')
    pipeline_description.add_step(step_7)
    curr_step_no += 1

    # Step 7: Integraying MultiAugmentation 
    if multi_aug == None:
        alg_python_path = 'd3m.primitives.autovideo.augmentation.meta_Sequential'
    else:
        alg_python_path = 'd3m.primitives.autovideo.augmentation.'+multi_aug
    step_7 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
    step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
    #for key, value in config.items():
    #    step_6.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
    step_7.add_output('produce')
    pipeline_description.add_step(step_7)
    curr_step_no += 1


    #Step 8: Video primitive
    alg_python_path = 'd3m.primitives.autovideo.recognition.' + algorithm
    step_8 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
    step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_7.index}.produce')
    step_8.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_4.index}.produce')
    # Add hyperparameters
    for key, value in config.items():
        step_8.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
    step_8.add_output('produce')
    pipeline_description.add_step(step_8)
    curr_step_no += 1

    #Step 9: Construct the predictions
    step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.autovideo.common.construct_predictions'))
    step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_8.index}.produce')
    step_9.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_2.index}.produce')
    step_9.add_output('produce')
    step_9.add_hyperparameter(name = 'use_columns', argument_type=ArgumentType.VALUE, data = [0,1])
    pipeline_description.add_step(step_9)
    curr_step_no += 1

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference=f'steps.{step_9.index}.produce')

    return pipeline_description


def run(args):
    # Set the logger path
    from autovideo.utils import set_log_path, logger
    set_log_path(args.log_path)

    train_table_path = os.path.join(args.data_dir, 'train.csv')
    train_media_dir = os.path.join(args.data_dir, 'media')
    target_index = 2

    from autovideo import fit, compute_accuracy_with_preds
    # Read the CSV file
    train_dataset = pd.read_csv(train_table_path)

    # Build pipeline based on configs
    # Here we can specify the hyperparameters defined in each primitive
    # The default hyperparameters will be used if not specified
    config = {
        "algorithm": args.alg,
        "load_pretrained": args.pretrained,
    }
    pipeline = test_augmentation(
            config, 
            augmentation_methods=["meta_ChannelShuffle", "blur_GaussianBlur", "flip_Fliplr", "imgcorruptlike_GaussianNoise"],
            multi_aug = "meta_Sometimes"
            )

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

