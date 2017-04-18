import numpy as np

def compute_ndcg(scores):
  """
  Computes the NDCG metric
  :param scores: numpy array
      A ranked list of (ordered) of actual relevance scores
  :return: double
      The NDCG metric
  """

  scores = np.array(scores)

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

  dcg = compute_dcg(scores)
  opt_dcg = compute_dcg(opt_scores) + eps

  return dcg / opt_dcg
