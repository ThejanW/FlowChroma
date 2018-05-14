import cv2
import numpy as np


def resize_pad_frame(img, size, pad_color=255, equal_padding=True):
    """
    Resize the frame,
    If image is a horizontal one first match the horizontal axis then resize vertical axis and fill the remaining
    with padding color, similar process for vertical images
    :param equal_padding:
    :param img: frame to be resized
    :param size: final frame size
    :param pad_color: color of tha padding
    :return: re-sized frame
    """
    h, w = float(img.shape[0]), float(img.shape[1])
    expected_height, expected_width = size

    # interpolation method
    if h > expected_height or w > expected_width:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h

    # compute scaling and pad sizing
    if aspect >= 1:  # horizontal image
        new_w = expected_width
        new_h = np.round(new_w / aspect).astype(int)
        if expected_height >= new_h:
            if equal_padding:
                pad_vert = (expected_height - new_h) / 2.0
                pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            else:
                pad_vert = (expected_height - new_h)
                pad_top, pad_bot = 0, pad_vert
                pad_left, pad_right = 0, 0
        else:
            new_h = expected_height
            new_w = np.round(new_h * aspect).astype(int)
            if equal_padding:
                pad_horz = (expected_width - new_w) / 2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_bot = 0, 0
            else:
                pad_horz = (expected_width - new_w)
                pad_left, pad_right = 0, pad_horz
                pad_top, pad_bot = 0, 0

    elif aspect < 1:  # vertical image
        new_h = expected_height
        new_w = np.round(new_h * aspect).astype(int)
        if expected_width >= new_w:
            if equal_padding:
                pad_horz = (expected_width - new_w) / 2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_bot = 0, 0
            else:
                pad_horz = (expected_width - new_w)
                pad_left, pad_right = 0, pad_horz
                pad_top, pad_bot = 0, 0
        else:
            new_w = expected_width
            new_h = np.round(new_w / aspect).astype(int)
            if equal_padding:
                pad_vert = (expected_height - new_h) / 2.0
                pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            else:
                pad_vert = (expected_height - new_h)
                pad_top, pad_bot = 0, pad_vert
                pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(pad_color,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img
