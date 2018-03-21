from os import listdir
from os.path import join, isfile

import cv2


class Sample:
    def __init__(self, source_dir, dest_dir, length=32, skip=1):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.length = length
        self.skip = skip

    def save_sliced_video(self, input_file, output_file):
        video = cv2.VideoCapture(input_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, size)

        count = 0
        while count < self.length:
            ret, frame = video.read()
            if not ret:
                break
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % self.skip != 0:
                continue
            out.write(frame)
            count += 1
        video.release()
        out.release()

    def slice_all(self):
        # iterate over each file in the source directory
        file_list = listdir(self.source_dir)
        print("Start sampling %d files" % len(file_list))
        count = 0
        for file_name in file_list:
            input_file = join(self.source_dir, file_name)
            new_file_name = "video_" + format(count, '05d') + ".avi"
            output_file = join(self.dest_dir, new_file_name)
            if isfile(input_file):
                self.save_sliced_video(input_file, output_file)
            count += 1
            if count % 10 == 0:
                print("Processed %d out of %d" % (count, len(file_list)))
        print("Sampling completed %d out of %d" % (count, len(file_list)))


if __name__ == '__main__':
    import argparse
    from dataset.utils.shared import dir_originals, dir_sampled, frames_per_video

    parser = argparse.ArgumentParser(
        description='Sample video files')
    parser.add_argument('-s', '--source-folder',
                        type=str,
                        metavar='FOLDER',
                        default=dir_originals,
                        dest='source',
                        help='use FOLDER as source of the videos')
    parser.add_argument('-o', '--output-folder',
                        type=str,
                        metavar='FILE',
                        default=dir_sampled,
                        dest='output',
                        help='use FILE as destination')
    parser.add_argument('-l', '--length',
                        default=frames_per_video,
                        type=int,
                        metavar='LENGTH',
                        dest='length',
                        help='use LENGTH as number of frames')
    parser.add_argument('-k', '--skip',
                        default=3,
                        type=int,
                        metavar='SKIP',
                        dest='skip',
                        help='use SKIP as number of frames between two selected frames')

    args = parser.parse_args()

    sample = Sample(args.source, args.output, args.length, args.skip)
    sample.slice_all()
