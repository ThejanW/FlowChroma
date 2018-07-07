from multiprocessing import Pool
from os import listdir
from os.path import join
from dataset.utils.shared import frames_per_video, dir_lab_records, dir_resnet_csv, dir_frame_resnet_records, dir_frame_lab_records
import numpy as np


def process_file(index):
    lab_record = np.load(join(dir_lab_records, "lab_record_" + format(index, '05d') + ".npy"))
    resnet_record = np.load(join(dir_resnet_csv, "resnet_record_" + format(index, "05d") + ".npy"))
    for i in range(frames_per_video):
        np.save(join(dir_frame_lab_records, "lab_record_" + format(index * frames_per_video + i, '08d')), lab_record[i])
        np.save(join(dir_frame_resnet_records,"resnet_record_" + format(index * frames_per_video + i, '08d')), resnet_record[i])
    print('Index ' + str(index) + ' done')


if __name__ == '__main__':
    lab_records = listdir(dir_lab_records)
    resnet_records = listdir(dir_resnet_csv)

    assert len(lab_records) == len(resnet_records)
    n = len(lab_records)

    file_indexes = [x for x in range(n)]
    pool = Pool(2)
    pool.map(process_file, file_indexes)


