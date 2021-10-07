'''
Copyright 2021 D3M Team
Copyright (c) 2021 DATA Lab at Texas A&M University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import os
import uuid
from urllib.parse import urlparse

import numpy as np

from d3m import container
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

__all__ = ('NumpyLoadingPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class NumpyLoaderPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which loads numpy for each video
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, "numpy_loader")),
            'version': '0.0.1',
            'name': "Load numpy for each video",
            'python_path': 'd3m.primitives.autovideo.common.numpy_loader',
            'source': {
                'name': 'TAMU DATALAB - Daochen Zha',
                'contact': 'mailto:daochen.zha@rice.edu',
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        
        location_base_uris = inputs.metadata.query_column(0)['location_base_uris']
        media_dir = urlparse(location_base_uris[0]).path[:-1]

        # Load numpy
        data = []
        for i in range(len(inputs)):
            numpy_path = os.path.join(media_dir, inputs.iloc[i, 0])
            data.append([np.load(numpy_path)])

        return base.CallResult(Outputs(data))
