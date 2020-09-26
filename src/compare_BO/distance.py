import numpy as np


def distance(p, q):
    """
    Cholesky for logdet
    """
    avg = (p + q) / 2

    log_det_avg = logdet(avg)
    log_det_p = logdet(p)
    log_det_q = logdet(q)

    JS = 0.5 * (log_det_avg - log_det_p) + 0.5 * (log_det_avg - log_det_q)
    JS = JS / 2

    d = 1 - np.exp(-JS)

    return d


def logdet(a):
    c = 0
    while c < 20:
        try:
            L = np.linalg.cholesky(a + 0.1 * c * np.eye(a.shape[0]))
            break
        except:
            c += 1
            continue

    logdet = 2 * np.sum(np.log(np.diag(L)))
    return logdet
