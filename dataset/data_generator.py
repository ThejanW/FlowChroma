import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for the Keras model
    """

    def __init__(self,
                 resnet_path,
                 lab_path,
                 file_ids,
                 batch_size=2,
                 time_steps=3,
                 h=240,
                 w=320,
                 shuffle=True):

        self.resnet_path = resnet_path
        self.lab_path = lab_path
        self.file_ids = file_ids

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.h = h
        self.w = w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.file_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        batch_file_ids = [self.file_ids[k] for k in indexes]

        # Generate data
        return self.__data_generation(batch_file_ids)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.file_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_file_ids):
        """
        Generates data containing batch_size samples
        """

        x = [np.empty((self.batch_size, self.time_steps, self.h, self.w, 1)),
             np.empty((self.batch_size, self.time_steps, 1000))]

        y = np.empty((self.batch_size, self.time_steps, self.h, self.w, 2))

        # Generate data
        for i, file_id in enumerate(batch_file_ids):
            lab_record = np.load('{0}/lab_record_{1}.npy'.format(self.lab_path, file_id))
            resnet_record = np.load('{0}/resnet_record_{1}.npy'.format(self.resnet_path, file_id))

            x[0][i, :, :, :, 0] = lab_record[:, :, :, 0]
            x[1][i, :, :] = resnet_record

            y[i, :, :, :, :] = lab_record[:, :, :, 1:]

        return x, y
