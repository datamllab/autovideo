""" Some utils to to process frames
"""
import sys
import os
import glob
from multiprocessing import Pool

import mmcv


def dump_frames(vid_item):
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
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, len(vr)))
            break
    print('{} done with {} frames'.format(vid_name, len(vr)))
    sys.stdout.flush()
    return True

def extract_frames(media_dir, ext, num_worker=8):
    out_dir = os.path.join(media_dir, 'frames')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fullpath_list = glob.glob(media_dir + '/*' + ext)
    done_list = [p.split('/')[-1] for p in glob.glob(out_dir + '/*')]
    _fullpath_list = []
    for p in fullpath_list:
        if os.path.splitext(os.path.basename(p))[0] not in done_list:
            _fullpath_list.append(p)
    fullpath_list = _fullpath_list
    if len(fullpath_list) == 0:
        return
    vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))
    pool = Pool(num_worker)
    pool.map(dump_frames, zip(
        fullpath_list, vid_list, range(len(vid_list)), [out_dir]*len(vid_list)))

def get_frame_list(media_dir, inputs, outputs=None, test_mode=False):
    """Returns the frame list with frame directory, #frames, label for each video for training data and frame directory, #frames for test data"""
    frame_root_dir = os.path.join(media_dir, 'frames')
    video_list = []
    for i in range(len(inputs)):
        video_name = os.path.splitext(inputs.iloc[i, 0])[0]
        frame_dir = os.path.join(frame_root_dir, video_name)
        num_frames = len(os.listdir(frame_dir))
        if test_mode == False:
            label = outputs.iloc[i,0]
            video_list.append((frame_dir, num_frames, label))
        else:
            video_list.append((frame_dir, num_frames))

    return video_list

