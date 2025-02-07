import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # initialize input attributes
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        # add slots/default values for attributes added later
        self.centroids = None
        self.MSE = float("inf")
        self.cluster = None

        # basic error handling
        # check param types
        if type(k) != int:
            raise ValueError('Invalid type used for k. Please try again with an int value.')
        if type(tol) != float:
            raise ValueError('Invalid type used for tol. Please try again with a float value.')
        if type(max_iter) != int:
            raise ValueError('Invalid type used for max_iter. Please try again with an int value.')
        # check number of clusters
        if k < 1:
            raise ValueError('Invalid value for k. Please enter a value greater than zero.')
    
    def _initialize_centroids(self, mat: np.ndarray):
        self.centroids = np.random.uniform(np.min(mat, axis=0),
                                    np.max(mat, axis=0),
                                    size = (self.k, mat.shape[1]))
        
    def _update_centroids(self, mat: np.ndarray, curr_groups: np.ndarray):
        # initialize update centroid array
        up_centers = np.zeros((self.k, mat.shape[1]))
        # update the centroid value for each curr_group based on points in curr_group
        for i in range(self.k):
            # recalculate center and calculate how much the value changed from the previous iteration
            new_cent = np.mean(mat[curr_groups[i]], axis=0)
            delta_cent = np.max(np.linalg.norm(self.centroids[i] - new_cent))

            # if largest change less than tol value, keep previous centroid, else update
            if delta_cent < self.tol:
                up_centers[i] = self.centroids[i]
            else:
                up_centers[i] = new_cent
        return up_centers


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # initialize k centroids
        self._initialize_centroids(mat)
        
        # while loop for running iterations, track iteration # using iter and break the loop when (whichever condition met first): 
            # 1. iter==max_iter 
            # or
            # 2. all distances pass the tol threshold 
        iter = 0
        while iter < self.max_iter:
            # gather current distances, MSE of dists, and indices of which centroid has min dist for each pt
            dists = cdist(mat, self.centroids, metric='euclidean')
            min_dist = np.argmin(dists, axis=1)

            # gather and store which points assigned to each centroid (curr_groups) 
            self.cluster = {i: np.where(min_dist==i)[0] for i in range(self.k)}
            # calculate/update centers54
            up_centers = self._update_centroids(mat, self.cluster)

            # check if any centers not updated (aka if all groups passed the tol threshold)--if so, break. else, update centroids and move to next iteration
            if np.allclose(self.centroids, up_centers, self.tol):
                break
            else:
                self.centroids = up_centers

            iter+=1
        # calculate final MSE and add to self
        self.MSE = np.mean((cdist(mat, self.centroids, metric='euclidean'))**2)


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.cluster is None:
            raise ValueError('Fit the model using fit() before running predict().')
        if self.centroids.shape[1] != mat.shape[1]:
            raise ValueError('Input mat must have the same number of features as the data used for model fitting.')
        
        # extract cluster labels from self.cluster dictionary
        clusters = np.full(mat.shape[0], -1)
        for clust, ind in self.cluster.items():
            clusters[ind] = clust
        return clusters

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.MSE

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError('Fit the model using fit() before running get_centroids().')
        return self.centroids
