import inspect
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from data_tools import load_datafold
from pointwise_ranker import PointwiseRanker
from ranking_metrics import ndcg, err
from utils import plot_confusion_matrix, Timer

"""
Usage: experiments.py [EXPERIMENT_ID]
      EXPERIMENT_ID - ID of experiment. Choose from the following list:
 logistic regression -> with scikit learn -> baseline                   (1)
                     -> with TensorFlow   -> baseline                   (2)
                                          -> reguralisation (8 values)  (3)
                                          -> no feature normalization   (4)
                                          -> class balancing            (5)
 multilayer perceptron with feature normalization                       (6)
"""

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_dir = os.path.dirname(os.path.dirname(current_dir))

DATA_TXT = root_dir + os.sep + 'data'
DATA = root_dir + os.sep + 'data' + os.sep + 'cache'
RESULTS_DIR = root_dir + os.sep + 'results'

FOLDS = [1, 2, 3, 4, 5]
N_FEATURES = 136
SEED = 0
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
    metrics = metrics_fn(query_relevance.tolist())
    avg_metrics += metrics
  return avg_metrics / uniqe_qids.shape[0]


def run_experiment(experiment_name, folds=[1], model_type='logreg_tf',
                   reguralisation=0.01, n_subset_features=None,
                   feature_normalization=True, class_balancing=True, epochs=10,
                   max_iter=200,
                   test_only=False):
  print('Running experiment ' + experiment_name)

  # do n-fold cross-validation and measure accuracy and nDCG on vali and test set and take the average
  # save results in file
  accuracy_vali_list = []
  ndcg_vali_list = []
  err_vali_list = []

  accuracy_test_list = []
  ndcg_test_list = []
  err_test_list = []

  training_times = []

  for fold in folds:
    print('Fold ' + str(fold))

    # load data and store as numpy arrays
    print('loading data')
    source_dir = DATA_TXT + os.sep + 'Fold' + str(fold)
    cache_dir = DATA + os.sep + 'Fold' + str(fold)
    X_train, y_train, qid_train, X_vali, y_vali, qid_vali, X_test, y_test, qid_test = load_datafold(source_dir, N_FEATURES, cache_dir=cache_dir)

    if n_subset_features is not None:
      print('Using subset of features: ' + str(n_subset_features))
      X_train = X_train[:,:n_subset_features]
      X_vali = X_vali[:,:n_subset_features]
      X_test = X_test[:,:n_subset_features]

    input_dim = X_train.shape[1]

    if class_balancing:
      class_weight = 'balanced'
    else:
      class_weight = None


    if model_type == 'logreg_sk':
      model = LogisticRegression(C=reguralisation, random_state=SEED,
                                max_iter=max_iter, n_jobs=-1, verbose=1,
                                solver='sag', multi_class='multinomial',
                                class_weight=class_weight)

    elif model_type == 'logreg_tf':
      model = PointwiseRanker(input_dim,
                              N_CLASSES,
                              C=reguralisation,
                              epochs=epochs,
                              model_type='logreg',
                              batch_size=256,
                              class_weight=class_weight,
                              model_dir=root_dir + os.sep + 'model' + os.sep + experiment_name)

    elif model_type == 'MLP':
      model = PointwiseRanker(input_dim,
                              N_CLASSES,
                              epochs=epochs,
                              model_type='MLP',
                              batch_size=256,
                              class_weight=class_weight,
                              model_dir=root_dir + os.sep + 'model' + os.sep + experiment_name)

    else:
      raise ValueError('Unsupported model type: ' + model_type)


    if feature_normalization:
      scaler = preprocessing.StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_vali = scaler.transform(X_vali)
      X_test = scaler.transform(X_test)


    # fit
    if not test_only:
      if model_type == 'logreg_sk':
        fit_fn = lambda: model.fit(X_train, y_train)

      elif model_type == 'logreg_tf' or model_type == 'MLP':
        fit_fn = lambda: model.fit(X_train, y_train, validation_data=(X_vali, y_vali))

      else:
        raise ValueError('Unsupported model type: ' + model_type)

      print('fitting data')
      timer = Timer()
      timer.start()
      fit_fn()
      training_times.append(timer.stop())
    else:
      # load pretrained model
      model.load_model()

    def evaluate_split(X, y, qid):
      predictions = model.predict(X)
      accuracy = metrics.accuracy_score(y, predictions)

      cm = confusion_matrix(y, predictions)
      classes = [str(c) for c in range(N_CLASSES)]
      fig = plt.figure()
      plot_confusion_matrix(cm, classes)
      os.makedirs(root_dir + os.sep + 'figures', exist_ok=True)

      # compute ndcg metrics
      ndcg_score = metrics_over_dataset(predictions, y, qid, ndcg)
      # compute err metrics
      err_score = metrics_over_dataset(predictions, y, qid, err)

      return accuracy, ndcg_score, err_score, fig

    # eval on vali
    accuracy_vali, ndcg_vali, err_vali, fig = evaluate_split(X_vali, y_vali, qid_vali)
    accuracy_vali_list.append(accuracy_vali)
    ndcg_vali_list.append(ndcg_vali)
    err_vali_list.append(err_vali)

    print()
    print("Validation accuracy: %f" % accuracy_vali)
    print("Validation nDCG: %f" % ndcg_vali)
    print("Validation ERR: %f" % err_vali)
    fig.savefig(root_dir + os.sep + 'figures' + os.sep + experiment_name + '-cm-vali.png')

    # eval on test set
    accuracy_test, ndcg_test, err_test, fig = evaluate_split(X_test, y_test, qid_test)
    accuracy_test_list.append(accuracy_test)
    ndcg_test_list.append(ndcg_test)
    err_test_list.append(err_test)

    print()
    print("Test accuracy: %f" % accuracy_test)
    print("Test nDCG: %f" % ndcg_test)
    print("Test ERR: %f" % err_test)
    fig.savefig(root_dir + os.sep + 'figures' + os.sep + experiment_name + '-cm-test.png')



  avg_accuracy_vali = np.mean(np.array(accuracy_vali_list))
  avg_ndcg_vali = np.mean(np.array(ndcg_vali_list))
  avg_err_vali = np.mean(np.array(err_vali_list))

  avg_accuracy_test = np.mean(np.array(accuracy_test_list))
  avg_ndcg_test = np.mean(np.array(ndcg_test_list))
  avg_err_test = np.mean(np.array(err_test_list))

  std_accuracy_vali = np.std(np.array(accuracy_vali_list))
  std_ndcg_vali = np.std(np.array(ndcg_vali_list))
  std_err_vali = np.std(np.array(err_vali_list))

  std_accuracy_test = np.std(np.array(accuracy_test_list))
  std_ndcg_test = np.std(np.array(ndcg_test_list))
  std_err_test = np.std(np.array(err_test_list))

  avg_training_time = np.mean(np.array(training_times))

  results  = 'Averages of %d-fold cross-validation: \n' % len(folds)
  results += 'split \t\t accuracy \t nDCG \t\t ERR \n'
  results += 'validation \t %.3f+-%.3f \t %.3f+-%.3f \t %.3f+-%.3f \n' % (avg_accuracy_vali, std_accuracy_vali, avg_ndcg_vali, std_ndcg_vali, avg_err_vali, std_err_vali)
  results += 'test \t\t %.3f+-%.3f \t %.3f+-%.3f \t %.3f+-%.3f \n' % (avg_accuracy_test, std_accuracy_test, avg_ndcg_test, std_ndcg_test, avg_err_test, std_err_test)
  results += '\n'
  results += 'Average training time: %.1f seconds \n' % (avg_training_time)
  results += '\n'

  print(results)

  os.makedirs(RESULTS_DIR, exist_ok=True)
  with open(RESULTS_DIR + os.sep + experiment_name + '-results.txt', 'a') as f:
    f.write(results)


def print_help():
  s  = 'Usage: experiments.py [EXPERIMENT_ID] \n'
  s += '       EXPERIMENT_ID - ID of experiment. Choose from the following list: \n'
  s += '        logistic regression -> with scikit learn -> baseline                   (1) \n'
  s += '\n'
  s += '                            -> with TensorFlow   -> baseline                   (2) \n'
  s += '                                                 -> reguralisation (8 values)  (3) \n'
  s += '                                                 -> no feature normalization   (4) \n'
  s += '                                                 -> class balancing            (5) \n'
  s += '\n'
  s += '        multilayer perceptron                                                  (6) \n'
  print(s)



if __name__ == "__main__":
  folds = FOLDS
  test_only = False

  if len(sys.argv) != 2:
    print_help()
    exit()

  experiment = int(sys.argv[1])
  if   experiment == 1:
    run_experiment(str(experiment) + '-lg_sk-baseline',
                   folds=folds,
                   model_type='logreg_sk',
                   reguralisation=0.00001,
                   feature_normalization=True,
                   class_balancing=False,
                   max_iter=10,
                   test_only=test_only)

  elif experiment == 2:
    run_experiment(str(experiment) + '-lg_tf-baseline',
                   folds=folds,
                   model_type='logreg_tf',
                   reguralisation=0.0,
                   feature_normalization=True,
                   class_balancing=False,
                   epochs=10,
                   test_only=test_only)

  elif experiment == 3:
    reguralisations = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for reguralisation in reguralisations:
      run_experiment(str(experiment) + '-lg_tf-reg-' + str(reguralisation),
                     folds=folds,
                     model_type='logreg_tf',
                     reguralisation=reguralisation,
                     feature_normalization=True,
                     class_balancing=False,
                     epochs=10,
                     test_only=test_only)

  elif experiment == 4:
    run_experiment(str(experiment) + '-lg_tf-no_feat_norm',
                   folds=folds,
                   model_type='logreg_tf',
                   reguralisation=0.0,
                   feature_normalization=False,
                   class_balancing=False,
                   epochs=10,
                   test_only=test_only)

  elif experiment == 5:
    run_experiment(str(experiment) + '-lg_tf-balanc',
                   folds=folds,
                   model_type='logreg_tf',
                   reguralisation=0.0,
                   feature_normalization=True,
                   class_balancing=True,
                   epochs=10,
                   test_only=test_only)

  elif experiment == 6:
    run_experiment(str(experiment) + '-MLP',
                   folds=folds,
                   model_type='MLP',
                   feature_normalization=True,
                   class_balancing=False,
                   epochs=20,
                   test_only=test_only)
