import os
import uuid
import shutil
import pathlib

import pandas as pd

from d3m.metadata.problem import TaskKeyword, PerformanceMetric
from d3m.metadata import base as metadata_base
from axolotl.utils import pipeline as pipeline_utils
from axolotl.utils import data_problem
from axolotl.backend.simple import SimpleRunner

from .frames_utils import dump_frames


def generate_classification_dataset_problem(df, target_index, media_dir):
    if not os.path.isabs(medVia_dir):
        media_dir = os.path.abspath(media_dir)
    dataset, problem = data_problem.generate_dataset_problem(df,
                                                             target_index=target_index,
                                                             media_dir=media_dir,
                                                             performance_metrics=[{'metric': PerformanceMetric.ACCURACY}],
                                                             task_keywords=[TaskKeyword.CLASSIFICATION,])

    return dataset, problem

def generate_dataset(df, target_index, media_dir):
    if not os.path.isabs(media_dir):
        media_dir = os.path.abspath(media_dir)
    dataset = data_problem.import_input_data(df,
                                             y=None,
                                             target_index=target_index,
                                             media_dir=media_dir)

    return dataset

def generate_classification_problem(dataset):
    problem = data_problem.generate_problem_description(dataset,
                                                        performance_metrics=[{'metric': PerformanceMetric.ACCURACY}],
                                                        task_keywords=[TaskKeyword.CLASSIFICATION,])

    return problem

def fit(train_dataset, train_media_dir, target_index, pipeline):
    train_dataset = generate_dataset(train_dataset, target_index, train_media_dir)
    problem = generate_classification_problem(train_dataset)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    # Fit
    pipeline_result = backend.fit_pipeline(problem, pipeline, [train_dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error

    # Fetch the runtime and dataaset metadata
    fitted_pipeline = {
        'runtime': backend.fitted_pipelines[pipeline_result.fitted_pipeline_id],
        'dataset_metadata': train_dataset.metadata
    }

    return pipeline_result.output, fitted_pipeline

def produce(test_dataset, test_media_dir, target_index, fitted_pipeline):
    test_dataset['label'] = -1
    test_dataset = generate_dataset(test_dataset, target_index, test_media_dir)
    test_dataset.metadata = fitted_pipeline['dataset_metadata']

    metadata_dict = test_dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    metadata_dict['location_base_uris'] = [pathlib.Path(os.path.abspath(test_media_dir)).as_uri()+'/']
    test_dataset.metadata = test_dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = fitted_pipeline['runtime']

    # Produce
    pipeline_result = backend.produce_pipeline(_id, [test_dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result.output

def fit_produce(train_dataset, train_media_dir, test_dataset, test_media_dir, target_index, pipeline):
    _, fitted_pipeline = fit(train_dataset, train_media_dir, target_index, pipeline)
    output = produce(test_dataset, test_media_dir, target_index, fitted_pipeline)

    return output

def produce_by_path(fitted_pipeline, video_path):
    tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
    frame_dir = os.path.join(tmp_dir, "frames")
    video_name = video_path.split('/')[-1]
    dump_frames((video_path, video_name, 0, frame_dir))

    dataset = {
        'd3mIndex': [0],
        'video': [video_name]
    }
    dataset = pd.DataFrame(data=dataset)

    # Produce
    predictions = produce(test_dataset=dataset,
                          test_media_dir=tmp_dir,
                          target_index=2,
                          fitted_pipeline=fitted_pipeline)
    
    shutil.rmtree(tmp_dir)

    return predictions




