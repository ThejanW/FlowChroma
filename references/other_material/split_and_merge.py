import os

import cv2
import glob

import numpy as np


def split_videos():
    for vid_path in ['deep_col_clrd', 'flowchroma_clrd']:
        all_vids = glob.glob('{0}/*.avi'.format(vid_path))
        for vid in all_vids:
            os.makedirs('results/{0}'.format(vid[:-4]), exist_ok=True)
            cap = cv2.VideoCapture(vid)
            vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                count += 1
                print('Writing to frame results/{0}/{1}.png'.format(vid[:-4], count))
                cv2.imwrite('results/{0}/{1}.png'.format(vid[:-4], count), frame)
                if count == vid_length:
                    cap.release()
                    break
    print('Splitting frames successful!')


def merge_frames():
    all_vids_1 = glob.glob('deep_col_clrd/*.avi')
    all_vids_2 = glob.glob('flowchroma_clrd/*.avi')
    # checking whether, both folders have same videos
    assert [vid[-15:] for vid in all_vids_1] == [vid[-15:] for vid in all_vids_2]

    for vid in [vid[-15:-4] for vid in all_vids_1]:
        n_frames_1 = len(glob.glob('results/flowchroma_clrd/{0}/*.png'.format(vid)))
        n_frames_2 = len(glob.glob('results/deep_col_clrd/{0}/*.png'.format(vid)))
        # checking whether both folders have same number of frames
        assert n_frames_1 == n_frames_2

        os.makedirs('results/merged/{0}'.format(vid), exist_ok=True)

        for frame in range(1, n_frames_1 + 1):
            print(
                'Merging results/flowchroma_clrd/{0}/{1}.png with results/deep_col_clrd/{0}/{1}.png'.format(vid, frame))
            flowchroma_img = 'results/flowchroma_clrd/{0}/{1}.png'.format(vid, frame)
            deep_col_img = 'results/deep_col_clrd/{0}/{1}.png'.format(vid, frame)
            merged_frame = np.concatenate((cv2.imread(flowchroma_img), cv2.imread(deep_col_img)), axis=1)
            cv2.imwrite('results/merged/{0}/{1}.png'.format(vid, frame), merged_frame)
    print('Merging frames successful!')


split_videos()
merge_frames()
