import glob
import re
import time
from os import listdir, remove
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from dataset.embedding import maybe_download_inception, prepare_image_for_inception, inception_resnet_v2_arg_scope, \
    inception_resnet_v2
from dataset.tfrecords import queue_single_images_from_folder, batch_operations
from dataset.utils.resize import resize_pad_frame
from dataset.utils.shared import resnet_input_height, resnet_input_width


class ResnetRecordCreator:
    def __init__(self, video_dir, image_dir, record_dir, checkpoint_source):
        self.video_dir = video_dir
        self.image_dir = image_dir
        self.record_dir = record_dir
        self.checkpoint_file = maybe_download_inception(checkpoint_source)

    def convert_all(self, file_list):
        """
        Convert videos in the source directory to images
        :return:
        """
        for file_name in file_list:
            input_file = join(self.video_dir, file_name)

            # self.convert_video_to_images(input_file, sample_count)
            sample_index = int(re.search("video_(.*).avi", file_name).group(1))
            self.convert_video_to_images(input_file, sample_index)
            print("Video %d converted to frames" % sample_index)

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
            frame = resize_pad_frame(frame, (resnet_input_width, resnet_input_height))
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
            self._run_session(sess, operations)

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

    def _run_session(self, sess, operations):
        """
        Run the whole reading -> extracting features -> writing to records
        pipeline in a TensorFlow session
        :param sess:
        :param operations:
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
        final_resnet_results = []
        try:
            record_count = 0

            while not coord.should_stop():
                output_file = join(self.record_dir, "resnet_record_" + format(record_count, "05d"))
                final_resnet_results += self._write_record(operations, sess)
                record_count += 1
                print("Record Count %d" % record_count)

        except tf.errors.OutOfRangeError:
            # The string_input_producer queue ran out of strings
            pass
        finally:
            # Ask the threads (filename queue) to stop.
            coord.request_stop()
            final_resnet_results = sorted(final_resnet_results)
            for chunk in self.chunks(final_resnet_results, frames_per_video):
                file_index = int(re.search("image_(.*)_", str(chunk[0][0])).group(1))
                output_file = join(self.record_dir, "resnet_record_" + format(file_index, "05d"))
                csv_intermediate = [x.reshape(1001) for _, x in chunk]
                csv_out = np.asarray(csv_intermediate)
                np.save(output_file, csv_out)
                print(output_file + "written to disk. ")
                self._examples_count += 1
            print('Finished writing {} images in {:.2f}s'
                  .format(self._examples_count, time.time() - start_time))

        # Wait for threads to finish.
        coord.join(threads)

    def _write_record(self, operations, sess):
        # The base queue_operation is [a, b, c]
        # The batched queue_operation is [[a1, a2], [b1,b2], [c1, c2]]
        # and not [[a1, b1, c1], [a2, b2, c3]]
        # The result will have the same structure as the batched operations
        results = sess.run(operations)
        results = sorted(zip(results[0], results[2]))
        return results

    def process_all(self):
        file_list = listdir(self.video_dir)
        file_list = sorted(file_list)

        for chunk in self.chunks(file_list, resnet_video_chunk_size):
            # clear temporary directory
            files = glob.glob(self.image_dir + "/*")
            for f in files:
                remove(f)
            # convert to images of the size 299x299
            self.convert_all(chunk)
            # pass those images to RestNet
            self.batch_all(resnet_batch_size)

            tf.reset_default_graph()
            print("Chunk completed")

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]


if __name__ == '__main__':
    from dataset.utils.shared import dir_resnet_images, dir_resnet_csv, dir_sampled, frames_per_video, \
        resnet_video_chunk_size, resnet_batch_size, checkpoint_url
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
    parser.add_argument('-c', '--checkpoint',
                        default=checkpoint_url,
                        type=str,
                        dest='checkpoint',
                        help='set the source for the trained inception '
                             'weights, can be the url, the archive or the '
                             'file itself (default: {}) '
                        .format(checkpoint_url))

    args = parser.parse_args()

    resnetRecordConverter = ResnetRecordCreator(args.source, args.temporary, args.output,
                                                args.checkpoint)

    resnetRecordConverter.process_all()
