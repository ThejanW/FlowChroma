from os import listdir
from os.path import join, isfile
import numpy as np
import cv2


class VideoResizer:
    def __init__(self, source_dir, dest_dir,equal_padding = False):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.equal_padding = equal_padding

    def resize__pad_frame(self, img, size, pad_color=0):
        """
        Resize the frame,
        If image is a horizontal one first match the horizontal axis then resize vertical axis and fill the remaining
        with padding color, similar process for vertical images
        :param img: frame to be resized
        :param size: final frame size
        :param pad_color: color of tha padding
        :return: re-sized frame
        """
        h, w = float(img.shape[0]),float(img.shape[1])
        expected_height, expected_width = size

        # interpolation method
        if h > expected_height or w > expected_width:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w / h

        # compute scaling and pad sizing
        if aspect > 1:  # horizontal image
            new_w = expected_width
            new_h = np.round(new_w / aspect).astype(int)
            if self.equal_padding:
                pad_vert = (expected_height - new_h) / 2.0
                pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            else:
                pad_vert = (expected_height - new_h)
                pad_top, pad_bot = 0, pad_vert
                pad_left, pad_right = 0, 0

        elif aspect < 1:  # vertical image
            new_h = expected_height
            new_w = np.round(new_h * aspect).astype(int)
            if self.equal_padding:
                pad_horz = (expected_width - new_w) / 2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_bot = 0, 0
            else:
                pad_horz = (expected_width - new_w)
                pad_left, pad_right = 0, pad_horz
                pad_top, pad_bot = 0, 0
        else:  # square image
            new_h, new_w = expected_height, expected_width
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(pad_color,
                                                  (list, tuple, np.ndarray)):  # color image but only one color provided
            pad_color = [pad_color] * 3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=pad_color)

        return scaled_img

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
            resized_frame = self.resize__pad_frame(frame, size,255)
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
    parser.add_argument('-p', '--equal-padding',
                        default=False,
                        type=bool,
                        metavar='PAD',
                        dest='equal_padding',
                        help='use PAD to determine distribution of padding')

    args = parser.parse_args()

    video_resizer = VideoResizer(args.source, args.output,args.equal_padding)
    video_resizer.resize_all((args.height, args.width))
