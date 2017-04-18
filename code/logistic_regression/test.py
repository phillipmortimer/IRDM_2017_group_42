import numpy as np
from logreg import *

predictions = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
y_vali = np.array([2, 1, 0, 0, 2, 1, 0, 1, 0, 0])
qid_vali = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

predictions = np.array([9, 10, 8, 7, 6, 5, 4, 3, 2, 1, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1])
y_vali = np.array([1, 2, 0, 0, 2, 1, 0, 1, 0, 0, 1, 2, 0, 0, 2, 1, 0, 1, 0, 0])
qid_vali = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

predictions = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
y_vali = np.array([2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
qid_vali = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

ndcg = metrics_over_dataset(predictions, y_vali, qid_vali, compute_ndcg)
print("NDCG: %f" % ndcg)
