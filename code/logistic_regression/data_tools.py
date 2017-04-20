import os

import numpy as np


def load_datafold(source_dir, n_features, cache_dir=None):
  os.makedirs(cache_dir, exist_ok=True)

  def load_datafile(filename):
    print('loading ' + filename)

    with open(filename, 'r') as f:
      lines = f.readlines()

    n = len(lines)
    y = np.zeros(n, dtype=np.int32)
    qid = np.zeros(n, dtype=np.int32)
    X = np.zeros((n, n_features), dtype=np.float32)

    for i, line in enumerate(lines):
      if i % 10000 == 0:
        print('processed %d/%d lines' % (i, n))
      y[i] = line[0]
      line_split = line.split()
      qid[i] = line_split[1].split(':')[1]
      featues = [t.split(':')[1] for t in line_split[2:]]
      X[i, :] = featues

    return X, y, qid

  def load_datasplit(split):
    split_txt = source_dir + os.sep + split + '.txt'
    X_file = cache_dir + os.sep + 'X_' + split + '.npy'
    y_file = cache_dir + os.sep + 'y_' + split + '.npy'
    qid_file = cache_dir + os.sep + 'qid_' + split + '.npy'
    if not os.path.isfile(X_file) or not os.path.isfile(y_file):
      X, y, qid = load_datafile(split_txt)
      if cache_dir is not None:
        np.save(X_file, X)
        np.save(y_file, y)
        np.save(qid_file, qid)
    else:
      X = np.load(X_file)
      y = np.load(y_file)
      qid = np.load(qid_file)
    y = y.astype(np.int32)
    return X, y, qid

  X_train, y_train, qid_train = load_datasplit('train')
  X_vali, y_vali, qid_vali = load_datasplit('vali')
  X_test, y_test, qid_test = load_datasplit('test')

  return X_train, y_train, qid_train, X_vali, y_vali, qid_vali, X_test, y_test, qid_test
