from typing import List, Optional
from collections import OrderedDict
import uuid

import numpy as np

from d3m.container import DataFrame as d3m_dataframe
from d3m.primitive_interfaces.base import Hyperparams
from d3m.metadata import base as metadata_base

Inputs = d3m_dataframe
Outputs = d3m_dataframe

def construct_primitive_metadata(module, name):
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, name)),
            'version': '0.0.1',
            "name": "Implementation of " + name,
            'python_path': 'd3m.primitives.autovideo.' + module + '.' + name,
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        },
    )
    return metadata

def wrap_predictions(inputs: Inputs, predictions: np.ndarray, primitive_name: str) -> Outputs:
    outputs = d3m_dataframe(predictions, generate_metadata=True)
    target_columns_metadata = _add_target_columns_metadata(outputs.metadata, primitive_name)
    outputs.metadata = _update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)

    return outputs

def _add_target_columns_metadata(outputs_metadata: metadata_base.DataMetadata, primitive_name):
    outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
    target_columns_metadata: List[OrderedDict] = []
    for column_index in range(outputs_length):
        column_name = "{0}{1}_{2}".format(primitive_name, 0, column_index)
        column_metadata = OrderedDict()
        semantic_types = set()
        semantic_types.add('https://metadata.datadrivendiscovery.org/types/Attribute')
        column_metadata['semantic_types'] = list(semantic_types)

        column_metadata["name"] = str(column_name)
        target_columns_metadata.append(column_metadata)

    return target_columns_metadata

def _update_predictions_metadata(inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                             target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
    outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

    for column_index, column_metadata in enumerate(target_columns_metadata):
        column_metadata.pop("structural_type", None)
        outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

    return outputs_metadata



def build_pipeline(config):
    """Build a pipline based on the config
    """
    default_config = {
        "transformation": [],
        "augmentation": [],
        "multi_aug": None,
        "algorithm": "tsn",
    }
    for key in config:
        default_config[key] = config[key]
    config = default_config

    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep
    algorithm = config.pop('algorithm', None)
    transformation = config.pop('transformation', None)
    transformation_methods = [transformation[i][0] for i in range(len(transformation))]
    augmentation = config.pop('augmentation', None)
    augmentation_methods = [augmentation[i][0] for i in range(len(augmentation))]
    if len(augmentation) > 0 and len(augmentation[0]) > 1:
        augmentation_configs = []
        for i in range(len(augmentation)):
            try:
                augmentation_configs.append(augmentation[i][1])
            except:
                augmentation_configs.append(None)
        #augmentation_configs = [augmentation[i][1] for i in range(len(augmentation))]
    else:
        augmentation_configs = None
    multi_aug = config.pop('multi_aug', 'meta_Sequential')

    if len(transformation) > 0 and len(transformation[0]) > 1:
        transformation_configs = [transformation[i][1] for i in range(len(transformation))]
    else:
        transformation_configs = None

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

    curr_step_no = int(f'{step_5.index}')
    for i in range(len(transformation_methods)):
        alg_python_path = 'd3m.primitives.autovideo.transformation.' + transformation_methods[i]
        step_transformation = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
        step_transformation.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        if transformation_configs[i] != None:
            for key in transformation_configs[i]:
                value = transformation_configs[i][key]
                step_transformation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_transformation.add_output('produce')
        pipeline_description.add_step(step_transformation)
        curr_step_no += 1

    # Step 6: Video Augmentation
    for i in range(len(augmentation_methods)):
        alg_python_path = 'd3m.primitives.autovideo.augmentation.' + augmentation_methods[i]
        step_augmentation = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
        step_augmentation.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        if augmentation_configs[i] != None:
            for key in augmentation_configs[i]:
                value = augmentation_configs[i][key]
                step_augmentation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_augmentation.add_output('produce')
        pipeline_description.add_step(step_augmentation)
        curr_step_no += 1

    # Step 7: Integrating MultiAugmentation 
    if multi_aug != None:
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
    step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
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
