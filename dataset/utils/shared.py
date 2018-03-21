import itertools
from os import makedirs
from os.path import expanduser, join

# Default folders
dir_root = join(expanduser('~'), 'flowchroma')
dir_originals = join(dir_root, 'original')
dir_sampled = join(dir_root, 'sampled')
dir_restnet_images = join(dir_root, 'resized_restnet_images')
dir_restnet_csv = join(dir_root, 'restnet_csv_records')
dir_lab_records = join(dir_root, 'lab_records')
dir_tfrecord = join(dir_root, 'tfrecords')
dir_checkpoints = join(dir_root, 'checkpoints')
frames_per_video = 32


def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)


def initialize():
    maybe_create_folder(dir_sampled)
    maybe_create_folder(dir_restnet_images)
    maybe_create_folder(dir_restnet_csv)
    maybe_create_folder(dir_lab_records)
    maybe_create_folder(dir_tfrecord)


initialize()
