from os import listdir, makedirs
from os.path import join

import cv2

class FrameExtractor:
    def __init__(self, input_dir, output_dir, type):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.type = type

    @staticmethod
    def write_frame_to_image(self, file):
        """
        Read content from AVI file and save frames as images
            :param file: video file
            :param output_file: output image file
        """
        video = cv2.VideoCapture(file)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        for i, frame in enumerate(frames):
            frame_name = join(self.output_dir, file.split(".")[0].split("/")[-1] + "_img_{:05d}" + '.' + self.type).format(i)
            cv2.imwrite(frame_name, frame)



    def write_all(self):
        makedirs(self.output_dir, exist_ok=True)
        
        file_list = []
        for file_name in listdir(self.input_dir):
            if file_name.endswith(".avi"):
                file_list.append(join(self.input_dir, file_name))
        file_list = sorted(file_list)

        print("Start processing %d files" % len(file_list))
        for i in range(len(file_list)):
            self.write_frame_to_image(self, file_list[i])
            if i % 10 == 0:
                print("Processed %d out of %d" % (i, len(file_list)))
        print("Saving images completed.")


if __name__ == '__main__':
    import argparse
    # from dataset.utils.shared import dir_sampled, dir_lab_records

    parser = argparse.ArgumentParser(
        description='Create images from video frames')
    parser.add_argument('-s', '--source-folder',
                        type=str,
                        required=True,
                        metavar='FOLDER',
                        default=None,
                        dest='source',
                        help='use FOLDER as the source of videos')
    parser.add_argument('-o', '--output-folder',
                        type=str,
                        required=True,
                        metavar='FILE',
                        default=None,
                        dest='output',
                        help='use FILE as destination')
    parser.add_argument('-t', '--type',
                        default='jpeg',
                        type=str,
                        metavar='TYPE',
                        dest='type',
                        help='use TYPE as image file type')

    args = parser.parse_args()
    frameExtractor = FrameExtractor(args.source, args.output, args.type)
    frameExtractor.write_all()