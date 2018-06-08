import pandas as pd
import os
from tflearn.data_utils import build_hdf5_image_dataset

TRAIN = 'training.csv'
#VAL = 'validation.csv'
TRAIN_TRIM = 'training_trim.csv'

FULL_COLS = ['file_path','face_x','face_y','face_width','face_height','facial_landmarks','expression','valence','arousal']
TRIM_COLS = ['file_path', 'face_x', 'face_y', 'face_width', 'face_height', 'expression']

training = pd.read_csv(os.environ['HOME'] + '/' + TRAIN)
#validation = pd.read_csv(VAL, names=FULL_COLS)

training.columns = FULL_COLS
training = training[TRIM_COLS]
training.to_csv(TRAIN_TRIM)

TRAIN_FILE = 'training_trim.csv'
DATASET_FILE = 'train_dataset.txt'
TRAIN_H5 = 'train.h5'
VAL_H5 = 'val.h5'
IMG_SIZE = 256
DATA_DIR = os.environ['HOME']

train = pd.read_csv(TRAIN_FILE)
# split into training and test


f = open(DATASET_FILE, 'w')
for idx, row in training.iterrows():
    file_path = os.environ['HOME'] + '/' + row['file_path'].split('/')[1]
    folder = os.environ['HOME'] + '/' + row['file_path'].split('/')[0]
    if not os.path.isfile(file_path): continue
    # create folder if needed
    # crop image
    # write logic to toss out poorly formatted images
    line_string = '{} {}\\n'.format(file_path, row['expression'])
    f.write(line_string)
f.close()

dataset_file = 'train_dataset_trim.txt'
build_hdf5_image_dataset(dataset_file, image_shape=(IMG_SIZE, IMG_SIZE), mode='file',
                         output_path=TRAIN_H5, normalize=True, categorical_labels=True)

build_hdf5_image_dataset(dataset_file, image_shape=(IMG_SIZE, IMG_SIZE), mode='file',
                         output_path=TRAIN_H5, normalize=True, categorical_labels=True)