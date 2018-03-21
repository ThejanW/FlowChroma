import time
from os import listdir
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from dataset.embedding import maybe_download_inception, prepare_image_for_inception, inception_resnet_v2_arg_scope, \
    inception_resnet_v2
from dataset.tfrecords import queue_single_images_from_folder, batch_operations
from dataset.utils.resize import resize_pad_frame


class RestnetRecordCreator:
    def __init__(self, video_dir, image_dir, record_dir, checkpoint_source):
        self.video_dir = video_dir
        self.image_dir = image_dir
        self.record_dir = record_dir
        self.checkpoint_file = maybe_download_inception(checkpoint_source)

    def convert_all(self):
        """
        Convert videos in the source directory to images
        :return:
        """
        sample_count = 0
        for file_name in listdir(self.video_dir):
            input_file = join(self.video_dir, file_name)
            self.convert_video_to_images(input_file, sample_count)
            sample_count += 1
            if sample_count % 10 == 0:
                print("%d videos complete writing to frames" % sample_count)

    def convert_video_to_images(self, input_file, sample_count):
        """
        convert a single video file to images
        :param input_file:
        :param sample_count:
        :return:
        """
        video = cv2.VideoCapture(input_file)
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = resize_pad_frame(frame, (299, 299))
            # output file name image_videoindex_frameindex.jpeg ex: image_00001_005.jpeg
            output_file = join(self.image_dir,
                               "image_" + format(sample_count, '05d') + "_" + format(frame_count, '03d') + ".jpeg")
            cv2.imwrite(output_file, frame)
            frame_count += 1

    def batch_all(self, examples_per_record):
        """
        run tf session to pass images through restnet
        :param examples_per_record:
        :return:
        """
        operations = self._create_operations(examples_per_record)
        with tf.Session() as sess:
            self._initialize_session(sess)
            self._run_session(sess, operations, examples_per_record)

    def _create_operations(self, examples_per_record):
        """
        Create the operations to read images from the queue and
        extract inception features
        :return: a tuple containing all these operations
        """
        # Create the queue operations
        image_key, image_tensor, _ = queue_single_images_from_folder(self.image_dir)

        # Build Inception Resnet v2 operations using the image as input
        # - from rgb to grayscale to loose the color information
        # - from grayscale to rgb just to have 3 identical channels
        # - from a [0, 255] int8 range to [-1,+1] float32
        # - feed the image into inception and get the embedding
        img_for_inception = tf.image.rgb_to_grayscale(image_tensor)
        img_for_inception = tf.image.grayscale_to_rgb(img_for_inception)
        img_for_inception = prepare_image_for_inception(img_for_inception)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            input_embedding, _ = inception_resnet_v2(img_for_inception,
                                                     is_training=False)

        operations = image_key, image_tensor, input_embedding

        return batch_operations(operations, examples_per_record)

    def _initialize_session(self, sess):
        """
        Initialize a new session to run the operations
        :param sess:
        :return:
        """

        # Initialize the the variables that we introduced (like queues etc.)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Restore the weights from Inception
        # (do not call a global/local variable initializer after this call)
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)

    def _run_session(self, sess, operations, examples_per_record):
        """
        Run the whole reading -> extracting features -> writing to records
        pipeline in a TensorFlow session
        :param sess:
        :param operations:
        :param examples_per_record:
        :return:
        """

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        self._examples_count = 0

        # These are the only lines where something happens:
        # we execute the operations to get the image, compute the
        # embedding and write everything in the TFRecord
        try:
            record_count = 0
            while not coord.should_stop():
                output_file = join(self.record_dir, "restnet_record_" + format(record_count, "05d"))
                self._write_record(examples_per_record, operations, sess, output_file)
                record_count += 1
        except tf.errors.OutOfRangeError:
            # The string_input_producer queue ran out of strings
            pass
        finally:
            # Ask the threads (filename queue) to stop.
            coord.request_stop()
            print('Finished writing {} images in {:.2f}s'
                  .format(self._examples_count, time.time() - start_time))

        # Wait for threads to finish.
        coord.join(threads)

    def _write_record(self, examples_per_record, operations, sess, output_file):
        # The base queue_operation is [a, b, c]
        # The batched queue_operation is [[a1, a2], [b1,b2], [c1, c2]]
        # and not [[a1, b1, c1], [a2, b2, c3]]
        # The result will have the same structure as the batched operations
        results = sess.run(operations)
        results = sorted(zip(results[0], results[2]))
        csv_intermediate = [x.reshape(1001) for _, x in results]
        csv_out = np.asarray(csv_intermediate)
        np.save(output_file, csv_out)
        print(output_file + "written to disk. ")


if __name__ == '__main__':
    from dataset.utils.shared import dir_restnet_images, dir_restnet_csv, dir_sampled, frames_per_video
    import argparse

    checkpoint_url = "~/imagenet/inception_resnet_v2_2016_08_30.ckpt"

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
                        default=dir_restnet_csv,
                        help='use FILE as destination')
    parser.add_argument('-p', '--temporary-folder',
                        type=str,
                        metavar='FILE',
                        dest='temporary',
                        default=dir_restnet_images,
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
    parser.add_argument('-c', '--checkpoint',
                        default=checkpoint_url,
                        type=str,
                        dest='checkpoint',
                        help='set the source for the trained inception '
                             'weights, can be the url, the archive or the '
                             'file itself (default: {}) '
                        .format(checkpoint_url))
    parser.add_argument('-d', '--equal-padding',
                        default=True,
                        type=bool,
                        metavar='PAD',
                        dest='equal_padding',
                        help='use PAD to determine distribution of padding')

    args = parser.parse_args()


    restnetRecordConverter = RestnetRecordCreator(args.source, args.temporary, args.output,
                                                  args.checkpoint)
    restnetRecordConverter.convert_all()
    print("All video converted to images")
    # batch size should be equal to number of frames in one video
    restnetRecordConverter.batch_all(frames_per_video)
