import numpy as np

def err(scores, g_max=4, k=10):
    """
    Computes the ERR metric at rank k

    Chapelle et al:
    Expected reciprocal rank for graded relevance
    http://dl.acm.org/citation.cfm?id=1646033

    :param scores: list
        A ranked list (ordered) of actual relevance scores
    :param max_g: int
        The maximum possible relevance score
    :param k: int, optional
        The rank at which the metric should be returned
    :return: double
        The ERR at k over the input lists
    """

    # Follow the algorithm pseudo-code (Algorithm 2)
    p = 1.0
    ERR = 0.0

    for i in range(k):
        # First map the relevance score to the probability of a relevance score
        R = (2.0 ** scores[i] - 1) / (2.0 ** g_max)
        ERR += p * R / (i + 1.0)
        p *= (1.0 - R)

    return ERR


def ndcg(scores, k=10):
    safe_k = min(len(scores), k)
    if k > len(scores):
        debug = True
    def dcg(rels, dcg_k):
        dcg_ = 0.0
        for i in range(dcg_k):
            if dcg_k > len(rels):
                debug = True
            dcg_ += (2.0 ** rels[i] - 1) / np.log2(i + 2.0)
        return dcg_

    unnorm_dcg = dcg(scores, safe_k)
    sorted_scores = np.flipud(np.sort(scores))
    best_dcg = dcg(sorted_scores, safe_k)
    eps = 0.000001

    return (unnorm_dcg + eps)/ (best_dcg + eps)


