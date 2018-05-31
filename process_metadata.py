import pandas as pd

TRAIN = 'training.csv'
VAL = 'validation.csv'
TRAIN_TRIM = 'training_trim.csv'
VAL_TRIM = 'validation_trim.csv'
FULL_COLS = ['file_path','face_x','face_y','face_width','face_height','facial_landmarks','expression','valence','arousal']
TRIM_COLS = ['file_path', 'face_x', 'face_y', 'face_width', 'face_height', 'expression']



def main():
  training = pd.read_csv(TRAIN, names=FULL_COLS)
  validation = pd.read_csv(VAL, names=FULL_COLS)
  training[TRIM_COLS].to_csv(TRAIN_TRIM)
  validation[TRIM_COLS].to_csv(VAL_TRIM)