from os import listdir
import cv2
from os.path import join, isfile, isdir


class VideoResizer:
    def __init__(self, source_dir, dest_dir):
        self.source_dir = source_dir
        self.dest_dir = dest_dir

    def resize_video(self, input_file, output_file, size):
        video = cv2.VideoCapture(input_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, size)

        # iterate through each frame
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # use built in resize function - need to be changed
            resized_frame = cv2.resize(frame, size)
            out.write(resized_frame)
        video.release()
        out.release()

    def resize_all(self, size):
        # iterate each file in the source directory
        for file_name in listdir(self.source_dir):
            input_file = join(self.source_dir, file_name)
            if isfile(input_file):
                self.resize_video(input_file, join(self.dest_dir, file_name.replace(".mp4", ".avi")), size)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Resize videos from')
    parser.add_argument('-s', '--source-folder',
                        type=str,
                        metavar='FOLDER',
                        dest='source',
                        help='use FOLDER as source of the videos')
    parser.add_argument('-o', '--output-folder',
                        type=str,
                        metavar='FILE',
                        dest='output',
                        help='use FILE as destination')
    parser.add_argument('-t', '--height',
                        default=244,
                        type=int,
                        metavar='HEIGHT',
                        dest='height',
                        help='use HEIGHT as height of frames')
    parser.add_argument('-w', '--width',
                        default=244,
                        type=int,
                        metavar='WIDTH',
                        dest='width',
                        help='use WIDTH as width of a frame')

    args = parser.parse_args()

    video_resizer = VideoResizer(args.source, args.output)
    video_resizer.resize_all((args.width, args.height))
