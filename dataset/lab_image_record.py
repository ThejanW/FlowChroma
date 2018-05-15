from os import listdir
from os.path import join

import cv2
import numpy as np

from dataset.utils.resize import resize_pad_frame
from skimage import color


class ImageRecord:
    def __init__(self, input_dir, output_dir, size, equal_padding):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.size = size
        self.equal_padding = equal_padding

    @staticmethod
    def write_to_csv(file, output_file, size, equal_padding):
        """
        Read content from AVI file and resize frames and convert frames to LAB color space
        :param file: video file
        :param output_file: LAB output file
        :param size: (width, height)
        :param equal_padding: True/False
        """
        video = cv2.VideoCapture(file)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = resize_pad_frame(frame, size, equal_padding=equal_padding)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            frame = color.rgb2lab(frame)
            frames.append(frame)
        frames = np.asarray(frames)

        # LAB layers should be brought to [-1, +1] region
        frames[:, :, :, 0] = np.divide(frames[:, :, :, 0], 50) - 1
        frames[:, :, :, 1] = np.divide(frames[:, :, :, 1], 128)
        frames[:, :, :, 2] = np.divide(frames[:, :, :, 2], 128)

        np.save(output_file, frames)

    def write_all(self):
        file_list = []
        for file_name in listdir(self.input_dir):
            file_list.append(join(self.input_dir, file_name))
        file_list = sorted(file_list)

        print("Start processing %d files" % len(file_list))
        for i in range(len(file_list)):
            self.write_to_csv(file_list[i], join(self.output_dir, "lab_record_" + format(i, '05d')), self.size,
                              self.equal_padding)
            if i % 10 == 0:
                print("Processed %d out of %d" % (i, len(file_list)))
        print("LAB conversion completed")


if __name__ == '__main__':
    import argparse
    from dataset.utils.shared import dir_sampled, dir_lab_records, default_nn_input_height, default_nn_input_width

    parser = argparse.ArgumentParser(
        description='Create lab records for video files')
    parser.add_argument('-s', '--source-folder',
                        type=str,
                        metavar='FOLDER',
                        default=dir_sampled,
                        dest='source',
                        help='use FOLDER as source of the videos')
    parser.add_argument('-o', '--output-folder',
                        type=str,
                        metavar='FILE',
                        default=dir_lab_records,
                        dest='output',
                        help='use FILE as destination')
    parser.add_argument('-t', '--height',
                        default=default_nn_input_height,
                        type=int,
                        metavar='HEIGHT',
                        dest='height',
                        help='use HEIGHT as height of frames')
    parser.add_argument('-w', '--width',
                        default=default_nn_input_width,
                        type=int,
                        metavar='WIDTH',
                        dest='width',
                        help='use WIDTH as width of a frame')
    parser.add_argument('-p', '--equal-padding',
                        default=True,
                        type=bool,
                        metavar='PAD',
                        dest='equal_padding',
                        help='use PAD to determine distribution of padding')

    args = parser.parse_args()
    imageRecord = ImageRecord(args.source, args.output, (args.height, args.width), args.equal_padding)
    imageRecord.write_all()
