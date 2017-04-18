import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from data_tools import load_datafold
from pointwise_ranker import PointwiseRanker
from ranking_metrics import compute_ndcg
from utils import plot_confusion_matrix


DATA_TXT = '../../data/MSLR-WEB30K'
DATA = '../../data/npy'

FOLDS = [1, 2]
C = 0.01
MAX_ITER = 200
N_FEATURES = 136
seed = 0
N_CLASSES = 5


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


if __name__ == "__main__":
  # load data and store as numpy arrays
  print('loading data')
  fold = 2
  source_dir = DATA_TXT + '/' + 'Fold' + str(fold)
  cache_dir = DATA + '/Fold' + str(fold)
  X_train, y_train, qid_train, X_vali, y_vali, qid_vali, X_test, y_test, qid_test = load_datafold(source_dir, N_FEATURES, cache_dir=cache_dir)

  # init log. reg.
  # lg = LogisticRegression(C=C, random_state=seed, max_iter=MAX_ITER, n_jobs=-1,
  #                         verbose=1, solver='lbfgs', multi_class='multinomial')

  input_dim = X_train.shape[1]
  lg = PointwiseRanker(input_dim, N_CLASSES, C=0.01, epochs=1,
                      model_type='MLP',
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

  # test on vali
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


  # test on test set
  predictions = lg.predict(X_test)
  score = metrics.accuracy_score(y_test, predictions)
  print()
  print("Accuracy: %f" % score)

  # compute ndcg metrics
  ndcg = metrics_over_dataset(predictions, y_test, qid_test, compute_ndcg)
  print("NDCG: %f" % ndcg)
