import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tf_logreg import LogisticRegressionTF
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools


DATA_TXT = '../../data/MSLR-WEB30K'
DATA = '../../data/npy'

FOLDS = [1, 2]
C = 0.01
MAX_ITER = 200
N_FEATURES = 136
seed = 0
N_CLASSES = 5


def load_datafile(filename):
  print('loading ' + filename)

  with open(filename, 'r') as f:
    lines = f.readlines()

  n = len(lines)
  y = np.zeros(n, dtype=np.int32)
  qid = np.zeros(n, dtype=np.int32)
  X = np.zeros((n, N_FEATURES), dtype=np.float32)

  for i, line in enumerate(lines):
    if i % 10000 == 0:
      print('processed %d/%d lines' % (i, n))
    y[i] = line[0]
    line_split = line.split()
    qid[i] = line_split[1].split(':')[1]
    featues = [t.split(':')[1] for t in line_split[2:]]
    X[i, :] = featues

  return X, y, qid


def preprocess_datafold(fold):
  os.makedirs(DATA, exist_ok=True)
  source_dir = DATA_TXT + '/' + 'Fold' + str(fold)
  save_dir = DATA + '/Fold' + str(fold)
  os.makedirs(save_dir, exist_ok=True)

  def preprocess_datasplit(split):
    split_txt = source_dir + '/' + split + '.txt'
    X_file = save_dir + '/X_' + split + '.npy'
    y_file = save_dir + '/y_' + split + '.npy'
    qid_file = save_dir + '/qid_' + split + '.npy'
    if not os.path.isfile(X_file) or not os.path.isfile(y_file):
      X, y, qid = load_datafile(split_txt)
      np.save(X_file, X)
      np.save(y_file, y)
      np.save(qid_file, qid)
    else:
      X = np.load(X_file)
      y = np.load(y_file)
      qid = np.load(qid_file)
    y = y.astype(np.int32)
    return X, y, qid

  X_train, y_train, qid_train = preprocess_datasplit('train')
  X_vali, y_vali, qid_vali = preprocess_datasplit('vali')
  X_test, y_test, qid_test = preprocess_datasplit('test')

  return X_train, y_train, qid_train, X_vali, y_vali, qid_vali, X_test, y_test, qid_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.

  Obtained from
  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
  """
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')



def compute_ndcg(scores):
  """
  Computes the NDCG metric
  :param scores: numpy array
      A ranked list of (ordered) of actual relevance scores
  :return: double
      The NDCG metric
  """

  def compute_dcg(scores):
    n = scores.shape[0]
    ranks = np.arange(n) + 1
    gain = 2 ** scores - 1
    discount = 1 / np.log2(ranks + 1)
    discounted_gain = gain * discount
    dcg = np.sum(discounted_gain)
    return dcg

  opt_scores = np.sort(scores)[::-1]

  eps = 0.000001

  dcg = compute_dcg(scores) + eps
  opt_dcg = compute_dcg(opt_scores) + eps

  return dcg / opt_dcg


def metrics_over_dataset(ranking, relevance, qids, metrics_fn):
  uniqe_qids = np.unique(qids)
  avg_metrics = 0.0
  # compute metrics for each quid
  for qid in uniqe_qids.tolist():
    query_idxs = qids == qid
    query_rankings = ranking[query_idxs]
    query_relevance = relevance[query_idxs]
    # sort by rankings
    sorted_idxs = np.argsort(query_rankings)[::-1]
    query_relevance = query_relevance[sorted_idxs]
    # compute metrics
    metrics = metrics_fn(query_relevance)
    avg_metrics += metrics
  return avg_metrics / uniqe_qids.shape[0]


# load data and store as numpy arrays
print('loading data')
X_train, y_train, qid_train, X_vali, y_vali, qid_vali, X_test, y_test, qid_test = preprocess_datafold(1)

# init log. reg.
# lg = LogisticRegression(C=C, random_state=seed, max_iter=MAX_ITER, n_jobs=-1,
#                         verbose=1, solver='lbfgs', multi_class='multinomial')

input_dim = X_train.shape[1]
lg = LogisticRegressionTF(input_dim, N_CLASSES, C=0.01, epochs=10,
                          batch_size=256,
                          class_weight=None,
                          model_dir='../../model/logreg-nb')

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_vali = scaler.transform(X_vali)

# load pretrained model
# lg.load_model()

# fit
print('fitting data')
# lg.fit(X_train, y_train)
lg.fit(X_train, y_train, validation_data=(X_vali, y_vali))


# test
predictions = lg.predict(X_vali)
score = metrics.accuracy_score(y_vali, predictions)
print()
print("Accuracy: %f" % score)

cm = confusion_matrix(y_vali, predictions)
classes = [str(c) for c in range(N_CLASSES)]
fig = plt.figure()
plot_confusion_matrix(cm, classes)
os.makedirs('../../figures', exist_ok=True)
fig.savefig('../../figures/' + 'cm.png')

# compute ndcg metrics
ndcg = metrics_over_dataset(predictions, y_vali, qid_vali, compute_ndcg)
print("NDCG: %f" % ndcg)
