#!/bin/bash
python3 dataset/utils/shared.py
python3 dataset/sampling.py
python3 dataset/lab_image_record.py
python3 dataset/resnet_record.py
