import numpy as np

def err(scores, g_max=4, k=10):
    """
    Computes the ERR metric at rank k

    :param scores: list
        A ranked list of (ordered) of actual relevance scores
    :param max_g: int
        The maximum possible relevance score
    :param k: int, optional
        The rank at which the metric should be returned
    :return: double
        The ERR at k over the input lists
    """

    if len(scores) > k:
        scores = scores[:k]

    # Follow the algorithm pseudo-code (Algorithm 2)
    p = 1
    ERR = 0

    for i in range(k):
        # First map the relevance score to the probability of a relevance score
        R = (2 ** scores[i] - 1) / (2 ** g_max)
        ERR += p * R / (i + 1)
        p *= (1 - R)

    return ERR