import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time

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


class Timer(object):
  def __init__(self, name=None, acc=False):
    self.name = name
    self.acc = acc
    self.total = 0.0

  def __enter__(self):
    self.start()

  def __exit__(self, type, value, traceback):
    self.stop()

  def start(self):
    self.tstart = time.time()

  def stop(self):
    self.total += time.time() - self.tstart
    if not self.acc:
      return self.reset()

  def reset(self):
    if self.name:
      print('[%s]' % self.name)
    elapsed = self.total
    print('Elapsed: %.4f' % elapsed)
    self.total = 0.0
    return elapsed
