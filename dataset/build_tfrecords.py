import os
import random
import sys
import threading
from collections import namedtuple
from datetime import datetime
from os.path import join

import numpy as np
import tensorflow as tf

from dataset.utils.shared import dir_lab_records, dir_resnet_csv, frames_per_video
from dataset.utils.shared import dir_tfrecord

tf.flags.DEFINE_integer("train_shards", 48,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 8,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")
tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to pre-process the videos.")
FLAGS = tf.flags.FLAGS

VideoMetaData = namedtuple("VideoMetaData", ["video_id", "lab_file", "resnet_file"])


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


class TFRecordBuilder:
    def __init__(self, resnet_folder, lab_folder):
        self.resnet_folder = resnet_folder
        self.lab_folder = lab_folder

    @staticmethod
    def _to_sequence_example(video):
        """Builds a SequenceExample proto for video lab file and embeddings.

        Args:
          video: An VideoMetadata object.

        Returns:
          A SequenceExample proto.
        """
        video_id = video.video_id
        resnet_file = video.resnet_file
        lab_file = video.lab_file
        resnet_record = np.load(resnet_file)
        lab_images = np.load(lab_file)
        L = lab_images[:, :, :, 0]
        A = lab_images[:, :, :, 1]
        B = lab_images[:, :, :, 2]
        embeddings = resnet_record

        # try:
        #     decoder.decode_jpeg(encoded_image)
        # except (tf.errors.InvalidArgumentError, AssertionError):
        #     print("Skipping file with invalid JPEG data: %s" % video.filename)
        #     return

        context = tf.train.Features(feature={
            "video/video_id": _int64_feature(video_id),
        })

        feature_lists = tf.train.FeatureLists(feature_list={
            "video/l_layer": _bytes_feature_list(L),
            "video/a_layer": _bytes_feature_list(A),
            "video/b_layer": _bytes_feature_list(B),
            "video/resnet_embedding": _bytes_feature_list(embeddings),
            "video/frame_ids": _int64_feature_list([x for x in range(frames_per_video)])
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

        return sequence_example

    @staticmethod
    def _process_video_data_files(thread_index, ranges, name, video_metadata, num_shards):
        """Processes and saves a subset of restnet embeddings and lab files as TFRecord files in one thread.

        Args:
          thread_index: Integer thread identifier within [0, len(ranges)].
          ranges: A list of pairs of integers specifying the ranges of the dataset to
            process in parallel.
          name: Unique identifier specifying the dataset.
          video_metadata: List of VideoMetadata.
          num_shards: Integer number of shards for the output files.
        """
        # Each thread produces N shards where N = num_shards / num_threads. For
        # instance, if num_shards = 128, and num_threads = 2, then the first thread
        # would produce shards [0, 64).
        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                                   num_shards_per_batch + 1).astype(int)
        num_videos_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
            output_file = os.path.join(dir_tfrecord, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)

            shard_counter = 0
            videos_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in videos_in_shard:
                video = video_metadata[i]

                sequence_example = TFRecordBuilder._to_sequence_example(video)
                if sequence_example is not None:
                    writer.write(sequence_example.SerializeToString())
                    shard_counter += 1
                    counter += 1

                if not counter % 1000:
                    print("%s [thread %d]: Processed %d of %d items in thread batch." %
                          (datetime.now(), thread_index, counter, num_videos_in_thread))
                    sys.stdout.flush()

            writer.close()
            print("%s [thread %d]: Wrote %d video data to %s" %
                  (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()
            shard_counter = 0
        print("%s [thread %d]: Wrote %d video data to %d shards." %
              (datetime.now(), thread_index, counter, num_shards_per_batch))
        sys.stdout.flush()

    @staticmethod
    def process_dataset(name, video_metadata, num_shards):
        """Processes a complete data set and saves it as a TFRecord.

        Args:
          name: Unique identifier specifying the dataset.
          video_metadata: List of VideoMetadata.
          num_shards: Integer number of shards for the output files.
        """
        # video_metadata = [VideoMetadata(video_id, lab_file, restnet_file)

        # Shuffle the ordering of videos. Make the randomization repeatable.
        random.seed(12345)
        random.shuffle(video_metadata)

        # Break the video_metadata into num_threads batches. Batch i is defined as
        # video_metadata[ranges[i][0]:ranges[i][1]].
        num_threads = min(num_shards, FLAGS.num_threads)
        spacing = np.linspace(0, len(video_metadata), num_threads + 1).astype(np.int)
        ranges = []
        threads = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()

        # Create a utility for decoding videos to run sanity checks.
        # decoder = VideoDecoder()

        # Launch a thread for each batch.
        print("Launching %d threads for spacings: %s" % (num_threads, ranges))
        for thread_index in range(len(ranges)):
            args = (thread_index, ranges, name, video_metadata, num_shards)
            t = threading.Thread(target=TFRecordBuilder._process_video_data_files, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print("%s: Finished processing all %d video resnet and lab pairs in data set '%s'." %
              (datetime.now(), len(video_metadata), name))

    @staticmethod
    def load_and_process_metadata(resnet_records_dir, lab_records_dir):
        """Loads video metadata from .npy files

        Args:

        Returns:
          A list of available LAB color files and restnet embedding files.
        """
        resnet_records = []
        for file_name in os.listdir(resnet_records_dir):
            resnet_records.append(join(resnet_records_dir, file_name))
        resnet_records = sorted(resnet_records)

        lab_records = []
        for file_name in os.listdir(lab_records_dir):
            lab_records.append(join(lab_records_dir, file_name))
        lab_records = sorted(lab_records)

        video_metadata = []
        for id in range(len(lab_records)):
            video_metadata.append(VideoMetaData(id, lab_records[id], resnet_records[id]))

        return video_metadata

    def process(self, train_size, val_size, test_size):
        metadata = tfRecordBuilder.load_and_process_metadata(self.resnet_folder, self.lab_folder)
        random.shuffle(metadata)

        train_cutoff = train_size
        val_cutoff = train_cutoff + val_size

        train_dataset = metadata[: train_cutoff]
        val_dataset = metadata[train_cutoff: val_cutoff]
        test_dataset = metadata[val_cutoff:]

        tfRecordBuilder.process_dataset("train", train_dataset, FLAGS.train_shards)
        tfRecordBuilder.process_dataset("val", val_dataset, FLAGS.val_shards)
        tfRecordBuilder.process_dataset("test", test_dataset, FLAGS.test_shards)


if __name__ == '__main__':
    import argparse
    from dataset.utils.shared import training_set_size, validation_set_size, test_set_size

    parser = argparse.ArgumentParser(
        description='Resize videos from')
    parser.add_argument('-r', '--resnet-records-folder',
                        type=str,
                        metavar='FOLDER',
                        default=dir_resnet_csv,
                        dest='resnet_folder',
                        help='use FOLDER as source of the videos')
    parser.add_argument('-l', '--lab-record-folder',
                        type=str,
                        metavar='LABFOLDER',
                        dest='lab_folder',
                        default=dir_lab_records,
                        help='use LABFOLDER  as source')
    parser.add_argument('-t', '--train',
                        default=training_set_size,
                        type=int,
                        metavar='TRAIN',
                        dest='train',
                        help='train set size')
    parser.add_argument('-v', '--validation',
                        default=validation_set_size,
                        type=int,
                        metavar='VALIDATION',
                        dest='validation',
                        help='use as validation test size')
    parser.add_argument('-e', '--test',
                        default=test_set_size,
                        type=int,
                        metavar='TEST',
                        dest='test',
                        help='use as test size')

    args = parser.parse_args()

    tfRecordBuilder = TFRecordBuilder(args.resnet_folder, args.lab_folder)
    tfRecordBuilder.process(args.train, args.validation, args.test)
