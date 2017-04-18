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
    p = 1
    ERR = 0

    for i in range(k):
        # First map the relevance score to the probability of a relevance score
        R = (2 ** scores[i] - 1) / (2 ** g_max)
        ERR += p * R / (i + 1)
        p *= (1 - R)

    return ERR