import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations
            s(i) = b(i) - a(i) / max(a(i), b(i))
                where:
                    - a(i) = avg distance to other points within cluster
                    - b(i) = min avg distance to points in another cluster

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # basic error handling
        if X.shape[0] != len(y):
            raise ValueError('Check your X and y matrices--number of items in X != number of items in y')
        
        unique_clusters = np.unique(y)
        silhouette_vals = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            # avg dist to pts within cluster
            a_i = np.mean(np.linalg.norm(X[i] - X[y == y[i]], axis=1))

            # min avg dist to points in another cluster
            b_i = np.min([np.mean(np.linalg.norm(X[i] - X[y == j], axis=1))
                          for j in unique_clusters if j != y[i]])
            
            # calculate sil val and add to output array
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouette_vals