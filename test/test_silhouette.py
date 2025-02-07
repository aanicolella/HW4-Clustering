from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from cluster.utils import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)

import pytest
import numpy as np
from sklearn.metrics import silhouette_samples

def assert_sil_ValErrors():
    # handle case where number of pts != number of labels
    with pytest.raises(ValueError):
        X_dummy, true_labels = make_clusters(k=5)
        y_dummy = np.random.randint(0, 5, size = (X_dummy.shape[0] + 42))
        scores = Silhouette.score(X_dummy, y_dummy)

def test_silhouette_score():
    # test scoring compared to sklearn.metrics.silhouette_samples
    X1, y1 = make_clusters()
    kmeans = KMeans(k=3)
    kmeans.fit(X1)
    y_pred = kmeans.predict(X1)

    sil = Silhouette()
    scores = sil.score(X1, y_pred)
    bench_scores = silhouette_samples(X1, y_pred)

    assert np.allclose(scores, bench_scores, atol=1e-2)

    # test for overlapping clusters
    X1, y1 = make_clusters(scale=2.5)
    kmeans = KMeans(k=3)
    kmeans.fit(X1)
    y_pred = kmeans.predict(X1)

    sil = Silhouette()
    scores = sil.score(X1, y_pred)
    bench_scores = silhouette_samples(X1, y_pred)

    assert np.allclose(scores, bench_scores, atol=1e-2)

