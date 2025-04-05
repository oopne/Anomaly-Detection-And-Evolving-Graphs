import numpy as np
import scipy.stats as sps
from scipy.spatial import KDTree
from sklearn.base import OutlierMixin, BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted


class GPDC(OutlierMixin, BaseEstimator):
    '''
    Generalised Pareto Distribution Classifier for anomaly detection
    '''
    def __init__(self, tail_size: int = 10, alpha: float = 0.01) -> None:
        '''
        :param tail_size: the number of upper order statistics to be used
        :param alpha: default threshold to be used for hypothesis tests
        '''
        self.tail_size = tail_size
        self.alpha = alpha

    def __shapes_and_neg_quantiles(self, neg_distances: np.ndarray) -> np.array:
        thresholds = neg_distances[:, -1]
        tails = neg_distances[:, :-1]
        
        shapes = np.log(-tails).mean(axis=1) - np.log(-thresholds)
        neg_quantiles = -sps.genpareto.ppf(1 - 1 / self.tail_size,
                                           c=shapes,
                                           loc=thresholds,
                                           scale=thresholds * shapes)
        return shapes, neg_quantiles

    def fit(self, X: np.ndarray, y: None = None) -> 'GPDC':
        '''
        :param X: matrix with training points from normal class
        '''
        self.is_fitted_ = True

        X = X if len(X.shape) == 2 else X.reshape((len(X), 1))
        X = validate_data(self, X)
        
        self._kd_tree = KDTree(X)

        # k + 1 upper order statistics
        neg_distances = -self._kd_tree.query(X, self.tail_size + 2)[0][:, 1:]
        shapes, neg_quantiles = self.__shapes_and_neg_quantiles(neg_distances)

        dim = X.shape[-1]
        self._sorted_dimshapes = np.sort(dim * shapes)
        self._sorted_neg_quantiles = np.sort(neg_quantiles)
        return self

    def __quantile(self, sorted_data: np.array, q: float) -> float:
        n = len(sorted_data)
        j = int(q * (n - 1) // 1)
        g = q * (n - 1) % 1
        return (1 - g) * sorted_data[j] + g * sorted_data[j + 1]

    def predict(self, X: np.ndarray, alpha: float | None = None) -> np.array:
        '''
        Returns 1 for normal points and -1 for abnormal ones.
        :param X: matrix with points to test for abnormality
        :param alpha: probability threshold for hypothesis tests
                      (if None, self.alpha is used)
        '''
        check_is_fitted(self)

        X = X if len(X.shape) == 2 else X.reshape((len(X), 1))
        X = validate_data(self, X, reset=False)

        # k + 1 upper order statistics
        neg_distances = -self._kd_tree.query(X, self.tail_size + 1)[0]
        shapes, neg_quantiles = self.__shapes_and_neg_quantiles(neg_distances)

        alpha = alpha or self.alpha
        # Bonferroni's correction for multiple testing
        dimshape_threshold = self.__quantile(self._sorted_dimshapes, 1 - alpha / 2)
        neg_quantile_threshold = self.__quantile(self._sorted_neg_quantiles,
                                                 1 - alpha / 2)

        dim = X.shape[-1]
        result = -np.ones(X.shape[0], dtype=int)
        result[(dim * shapes < dimshape_threshold) &
               (neg_quantiles < neg_quantile_threshold)] = 1
        return result


class DiscreteGPDC(OutlierMixin, BaseEstimator):
    '''
    GPDC modification to work with discrete distributions.
    '''
    def __init__(self, tail_size: int = 10, alpha: float = 0.01) -> None:
        '''
        :param tail_size: the number of upper order statistics to be used
        :param alpha: default threshold to be used for hypothesis tests
        '''
        self.tail_size = tail_size
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: None = None) -> 'DiscreteGPDC':
        '''
        :param X: matrix with training points from normal class
        '''
        self._is_fitted = True

        X = X if len(X.shape) == 2 else X.reshape((len(X), 1))
        X = validate_data(self, X)

        self._dataset = set(tuple(row) for row in X)
        self._gpdc = GPDC(self.tail_size, self.alpha).fit(np.array(list(self._dataset)))

        return self

    def predict(self, X: np.ndarray, alpha: float | None = None) -> np.array:
        '''
        Returns 1 for normal points and -1 for abnormal ones.
        :param X: matrix with points to test for abnormality
        :param alpha: probability threshold for hypothesis tests
                      (if None, self.alpha is used)
        '''
        check_is_fitted(self)
        
        X = X if len(X.shape) == 2 else X.reshape((len(X), 1))
        X = validate_data(self, X, reset=False)

        result = np.zeros(X.shape[0], dtype=int)
        exact_matches = np.array([tuple(row) in self._dataset for row in X])
        result[exact_matches] = 1

        print(exact_matches.sum())

        if (~exact_matches).sum() > 0:
            result[~exact_matches] = self._gpdc.predict(X[~exact_matches])

        return result