""" Some utils to to process frames
"""
import os


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

