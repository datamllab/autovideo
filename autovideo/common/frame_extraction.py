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
import glob
import logging
import sys
import uuid
from multiprocessing import Pool
from urllib.parse import urlparse

import mmcv
import os

from d3m import container
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

__all__ = ('ExtractFramesPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    num_worker = hyperparams.Constant[int](
        default=8,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class ExtractFramesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which extracts frames from media dir based on video extension.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, "extract_frames")),
            'version': '0.0.1',
            'name': "Extracts frame by video extension",
            'python_path': 'd3m.primitives.autovideo.common.extract_frames',
            'source': {
                'name': 'TAMU DATALAB - Yi-Wei Chen',
                'contact': 'mailto:yiwei_chen@tamu.edu',
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        num_worker = self.hyperparams['num_worker']

        location_base_uris = inputs.metadata.query_column(0)['location_base_uris']
        media_dir = urlparse(location_base_uris[0]).path[:-1]
        ext = inputs.iloc[0, 0].split('.')[-1]

        out_dir = os.path.join(media_dir, 'frames')
        os.makedirs(out_dir, exist_ok=True)

        fullpath_list = glob.glob(media_dir + '/*' + ext)
        done_list = [p.split('/')[-1] for p in glob.glob(out_dir + '/*')]
        _fullpath_list = []
        for p in fullpath_list:
            if os.path.splitext(os.path.basename(p))[0] not in done_list:
                _fullpath_list.append(p)
        fullpath_list = _fullpath_list
        if len(fullpath_list) != 0:
            vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))
            pool = Pool(num_worker)
            pool.map(self._dump_frames, zip(
                fullpath_list, vid_list, range(len(vid_list)), [out_dir] * len(vid_list)))

        outputs = self._get_frame_list(media_dir=media_dir, inputs=inputs)
        return base.CallResult(outputs)

    def _dump_frames(self, vid_item):
        """ Dump frames for one video
        """
        full_path, vid_path, vid_id, out_dir = vid_item
        vid_name = vid_path.split('.')[0]
        out_full_path = os.path.join(out_dir, vid_name)
        if not os.path.exists(out_full_path):
            os.makedirs(out_full_path)
        vr = mmcv.VideoReader(full_path)
        for i in range(len(vr)):
            if vr[i] is not None:
                mmcv.imwrite(
                    vr[i], '{}/img_{:05d}.jpg'.format(out_full_path, i + 1))
            else:
                break
        logger.info('{} done with {} out of {} frames'.format(vid_name, i, len(vr)))
        sys.stdout.flush()
        return True

    def _get_frame_list(self, media_dir, inputs):
        """Returns the frame list with frame directory, #frames"""
        frame_root_dir = os.path.join(media_dir, 'frames')
        video_list = []
        for i, row in inputs.iterrows():
            video_name = os.path.splitext(row["video"])[0]
            frame_dir = os.path.join(frame_root_dir, video_name)
            num_frames = len(os.listdir(frame_dir))
            video_list.append((frame_dir, num_frames))

        return Outputs(video_list, columns=["frame_dir", "num_frames"])
