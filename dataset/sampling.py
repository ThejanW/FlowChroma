from os import listdir
from os.path import join, isfile, isdir
import numpy as np
import cv2
import tensorflow as tf


class Sample:
    def __init__(self, source_dir, dest_file, length=32, skip=1):
        self.source_dir = source_dir
        self.dest_file = dest_file
        self.length = length
        self.skip = skip

    def slice_video(self, input_file):
        frame_list_l = []
        frame_list_a = []
        frame_list_b = []
        video = cv2.VideoCapture(input_file)
        count = 0

        while count < self.length:
            ret, frame = video.read()
            if not ret:
                break
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % self.skip != 0:
                continue

            # convert to LAB color space using OpenCV
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

            frame_list_l.append(lab_frame[:, :, 0])
            frame_list_a.append(lab_frame[:, :, 1])
            frame_list_b.append(lab_frame[:, :, 2])
            count += 1

        # return L and A,B separately
        return np.asarray(frame_list_l), np.asarray([frame_list_a, frame_list_b])

    def slice_all(self):
        batch_x = []
        batch_y = []
        # iterate over each file in the source directory
        for file_name in listdir(self.source_dir):
            input_file = join(self.source_dir, file_name)
            if isfile(input_file):
                X, Y = self.slice_video(input_file)
                batch_x.append(X)
                batch_y.append(Y)
        return np.asarray(batch_x), np.asarray(batch_y)

    def np_to_tfrecords(self, X, Y, file_path_prefix, verbose=True):
        """
        Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
        For supervised learning, feed training inputs to X and training labels to Y.
        For unsupervised learning, only feed training inputs to X, and feed None to Y.
        The length of the first dimensions of X and Y should be the number of samples.

        Parameters
        ----------
        X : numpy.ndarray of rank 2
            Numpy array for training inputs. Its dtype should be float32, float64, or int64.
            If X has a higher rank, it should be rshape before fed to this function.
        Y : numpy.ndarray of rank 2 or None
            Numpy array for training labels. Its dtype should be float32, float64, or int64.
            None if there is no label array.
        file_path_prefix : str
            The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
        verbose : bool
            If true, progress is reported.

        Raises
        ------
        ValueError
            If input type is not float (64 or 32) or int.

        """

        def _dtype_feature(ndarray):
            """match appropriate tf.train.Feature class with dtype of ndarray. """
            assert isinstance(ndarray, np.ndarray)
            dtype_ = ndarray.dtype
            if dtype_ == np.float64 or dtype_ == np.float32:
                return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
            elif dtype_ == np.int64:
                return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
            else:
                raise ValueError("The input should be numpy ndarray. \
                                   Instaed got {}".format(ndarray.dtype))

        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2  # If X has a higher rank,
        # it should be rshape before fed to this function.
        assert isinstance(Y, np.ndarray) or Y is None

        # load appropriate tf.train.Feature class depending on dtype
        dtype_feature_x = _dtype_feature(X)
        if Y is not None:
            assert X.shape[0] == Y.shape[0]
            assert len(Y.shape) == 2
            dtype_feature_y = _dtype_feature(Y)

            # Generate tfrecord writer
        result_tf_file = file_path_prefix + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(result_tf_file)
        if verbose:
            print ("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

        # iterate over each sample,
        # and serialize it as ProtoBuf.
        for idx in range(X.shape[0]):
            x = X[idx]
            if Y is not None:
                y = Y[idx]

            d_feature = {}
            d_feature['X'] = dtype_feature_x(x)
            if Y is not None:
                d_feature['Y'] = dtype_feature_y(y)

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

        if verbose:
            print ("Writing {} done!".format(result_tf_file))

    def process(self, length=32, skip=1):
        X, Y = self.slice_all()
        # problem: for 8 bit images
        # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        x_reshaped = np.divide(X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]), 50.0) - 1
        y_reshaped = np.divide(Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2] * Y.shape[3] * Y.shape[4]), 128)
        self.np_to_tfrecords(x_reshaped, y_reshaped, self.dest_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a tfrecord for data set')
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
    parser.add_argument('-l', '--length',
                        default=32,
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
    sample.process()
