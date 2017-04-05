import numpy as np
import os
from  sklearn.linear_model import LogisticRegression

DATA_TXT = '/home/tom/Files/Projects/UCL/irdm/MSLR-WEB10K'
DATA = '/home/tom/Files/Projects/UCL/irdm/data'
FOLDS = [1, 2]
C = 1.0
MAX_ITER = 100
N_FEATURES = 136
seed = 0


def load_datafile(filename):
  print('loading ' + filename)

  with open(filename, 'r') as f:
    lines = f.readlines()

  n = len(lines)
  y = np.zeros(n, dtype=np.float32)
  X = np.zeros((n, N_FEATURES), dtype=np.float32)

  for i, line in enumerate(lines):
    if i % 10000 == 0:
      print('processed %d/%d lines' % (i, n))
    y[i] = line[0]
    featues = [t.split(':')[1] for t in line.split()[2:]]
    X[i, :] = featues

  return X, y


def preprocess_data():
  os.makedirs(DATA, exist_ok=True)
  for fold in FOLDS:
    source_dir = DATA_TXT + '/' + 'Fold' + str(fold) + '/'
    save_dir = DATA + '/Fold' + str(fold)
    os.makedirs(save_dir, exist_ok=True)

    train_txt = source_dir + 'train.txt'
    X, y = load_datafile(train_txt)
    np.save(save_dir + '/X_train', X)
    np.save(save_dir + '/y_train', y)

    vali_txt = source_dir + 'vali.txt'
    X, y = load_datafile(vali_txt)
    np.save(save_dir + '/X_vali', X)
    np.save(save_dir + '/y_vali', y)

    test_txt = source_dir + 'test.txt'
    X, y = load_datafile(test_txt)
    np.save(save_dir + '/X_vali', X)
    np.save(save_dir + '/y_vali', y)


# load data and store as numpy arrays
preprocess_data()

LogisticRegression(C=C, random_state=seed, max_iter=MAX_ITER, n_jobs=1)
