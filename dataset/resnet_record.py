import re
from os import listdir
from os.path import join

import cv2
import numpy as np

from dataset.utils.inception_utils import inception_resnet_v2_predict
from dataset.utils.resize import resize_pad_frame
from dataset.utils.shared import resnet_input_height, resnet_input_width


class ResnetRecordCreator:
    def __init__(self, video_dir, image_dir, record_dir):
        self.video_dir = video_dir
        self.image_dir = image_dir
        self.record_dir = record_dir

    def convert_all(self, file_list):
        """
        Convert videos in the source directory to images
        :return:
        """
        video_frames = []
        video_indexes = []
        for file_name in file_list:
            input_file = join(self.video_dir, file_name)

            # self.convert_video_to_images(input_file, sample_count)
            sample_index = int(re.search("video_(.*).avi", file_name).group(1))
            resized_frames = self.convert_video_to_images(input_file)

            assert len(resized_frames) == frames_per_video

            video_indexes.append(sample_index)
            video_frames += resized_frames
            print("Video %d converted to frames" % sample_index)
        return [video_indexes, np.asarray(video_frames)]

    def convert_video_to_images(self, input_file):
        """
        convert a single video file to images
        :param input_file:
        :param sample_count:
        :return:
        """
        video = cv2.VideoCapture(input_file)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = resize_pad_frame(frame, (resnet_input_height, resnet_input_width))
            gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_scale_frame_colored = cv2.cvtColor(gray_scale_frame, cv2.COLOR_GRAY2RGB)
            frames.append(gray_scale_frame_colored)
        return frames

    def predict_all(self, video_details):
        video_indexes, video_frames = video_details
        predictions = inception_resnet_v2_predict(video_frames)
        self.write_files(video_indexes, predictions)

    def write_files(self, video_indexes, predictions):
        for i in range(len(video_indexes)):
            frames_start = i * frames_per_video
            frames_end = frames_start + frames_per_video
            file_index = video_indexes[i]
            file_content = predictions[frames_start:frames_end]
            output_file = join(self.record_dir, "resnet_record_" + format(file_index, "05d"))
            np.save(output_file, file_content)

    def process_all(self):
        file_list = listdir(self.video_dir)
        file_list = sorted(file_list)

        for chunk in self.chunks(file_list, resnet_video_chunk_size):
            # convert to images of the size 299x299
            video_details = self.convert_all(chunk)
            # pass those images to RestNet
            self.predict_all(video_details)

            print("Chunk completed")

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]


if __name__ == '__main__':
    from dataset.utils.shared import dir_resnet_images, dir_resnet_csv, dir_sampled, frames_per_video, \
        resnet_video_chunk_size, checkpoint_url
    import argparse

    parser = argparse.ArgumentParser(
        description='Resize videos from')
    parser.add_argument('-s', '--source-folder',
                        type=str,
                        metavar='FOLDER',
                        default=dir_sampled,
                        dest='source',
                        help='use FOLDER as source of the videos')
    parser.add_argument('-o', '--output-folder',
                        type=str,
                        metavar='FILE',
                        dest='output',
                        default=dir_resnet_csv,
                        help='use FILE as destination')
    parser.add_argument('-p', '--temporary-folder',
                        type=str,
                        metavar='FILE',
                        dest='temporary',
                        default=dir_resnet_images,
                        help='use FILE as destination')

    args = parser.parse_args()

    resnetRecordConverter = ResnetRecordCreator(args.source, args.temporary, args.output)

    resnetRecordConverter.process_all()
