import numpy as np
import scipy.stats as sps


def hill_estimator(sample: np.ndarray) -> np.ndarray:
    '''
    Calculate Hill's estimates of EVI for k = 1, ..., (number of positive observations) - 1
    '''
    data = sample[sample > 0]
    if not np.all(data[:-1] >= data[1:]):
        data[::-1].sort()
    log_data = np.log(data)

    return np.cumsum(log_data)[:-1] / np.arange(1, len(log_data)) - log_data[1:]


def generalised_jackknife_estimator(sample: np.ndarray,
                                    hill_estims: np.ndarray | None = None) -> np.ndarray:
    '''
    Calculate Generalised Jackknife estimates of EVI for k = 1, ..., (number of positive observations) - 1
    :param hill_estims: Hill's estimates of EVI
    '''
    data = sample[sample > 0]
    if not np.all(data[:-1] >= data[1:]):
        data[::-1].sort()
    log_data = np.log(data)

    if hill_estims is None:
        hill_estims = hill_estimator(data)

    nums = np.arange(1, len(log_data))
    sums = np.cumsum(log_data)[:-1]
    sums2 = np.cumsum(log_data**2)[:-1]

    M = (sums2 - 2 * sums * log_data[1:] + log_data[1:]**2 * nums) / nums

    return M / hill_estims - hill_estims


def eye_ball_tail_size(estimates: np.ndarray, w_proportion: float = 0.01,
                       h: float = 0.9, eps: float = 0.3) -> int:
    '''
    Eye-Ball selection of k for tail index estimation
    :param estimates: tail index estimates for different k
    '''
    w = int(len(estimates) * w_proportion)

    for k in range(1, len(estimates) - w):
        diffs = estimates[k + 1:k + w + 1] - estimates[k]
        if np.sum((diffs > -eps) & (diffs < eps)) / w > h:
            return k + 1


def samsee(sample: np.ndarray,
           hill_estims: np.ndarray | None = None,
           gj_estims: np.ndarray | None = None) -> np.ndarray:
    '''
    Calculates SAMSEE(k) for k = 1, ..., K*
    :param hill_estims: Hill's estimates of EVI
    :param gj_estims: Generalised Jackknife estimates of EVI
    '''
    data = sample[sample > 0]
    if not np.all(data[:-1] >= data[1:]):
        data[::-1].sort()
    log_data = np.log(data)

    if hill_estims is None:
        hill_estims = hill_estimator(data)

    if gj_estims is None:
        gj_estims = generalised_jackknife_estimator(data, hill_estims)

    gamma_V = (gj_estims + hill_estims) / 2

    hill_prefs = np.concatenate([[0], np.cumsum(hill_estims)])
    b = lambda k, K: (hill_prefs[K] - hill_prefs[k - 1]) / (K - k + 1) - hill_prefs[K] / K
    
    N = min(len(data), 4 * int(len(data)**0.5)) - 1

    AD = np.zeros(N)
    for K in range(1, N):
        b_arr = np.array([b(k, K) for k in range(1, K + 1)])
        AD[K] = ((gamma_V[:K] + b_arr - hill_estims[:K])**2).sum() / K

    deriv = lambda K, i: np.abs((AD[K] - AD[K + i]) / i)
    func = lambda K: deriv(K, -2) + deriv(K, -1) + deriv(K, 1) + deriv(K, 2)
    
    max_K = 3
    for K in range(3, N - 3):
        if func(K) < func(max_K):
            max_K = K

    grid = np.arange(max_K) + 1
    b_arr = np.array([b(k, max_K) for k in grid])

    return gj_estims[max_K - 1]**2 / grid + 4 * b_arr**2


def samsee_tail_size(sample: np.ndarray,
                     hill_estims: np.ndarray | None = None,
                     gj_estims: np.ndarray | None = None) -> int:
    '''
    Calculates k_{SAMSEE} for Generalised Jackknife estimator
    :param hill_estims: Hill's estimates of EVI
    :param gj_estims: Generalised Jackknife estimates of EVI
    '''
    return np.argmin(samsee(sample, hill_estims, gj_estims)) + 1


def tail_index_and_tail_size(sample: np.ndarray) -> tuple[float, int]:
    '''
    Returns optimal Hill's estimate of EVI and SAMSEE tail size
    '''
    data = sample[sample > 0]
    data[::-1].sort()

    hill_estims = hill_estimator(data)
    gj_estims = generalised_jackknife_estimator(data, hill_estims)
    tail_size = samsee_tail_size(data, hill_estims, gj_estims)

    return 1 / hill_estims[tail_size - 1], tail_size


def phillips_loretan_pvalue(alpha1: float, k1: float, alpha2: float, k2: float) -> float:
    '''
    Calculates Phillips-Loretan test pvalue
    '''
    stat = k1 * (alpha1 / alpha2 - 1)**2 / ((alpha1 / alpha2)**2 + (k1 / k2)**2)
    return sps.chi2.sf(x=stat, df=1)