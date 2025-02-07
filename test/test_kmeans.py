from cluster.kmeans import KMeans
from cluster.utils import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
import pytest
import numpy as np

def assert_km_ValueErrors():
    # init tests
    # invalid input types
    with pytest.raises(ValueError):
        kmeans = KMeans(k='5', tol=0.01, max_iter=100)
    with pytest.raises(ValueError):
        kmeans = KMeans(k=5, tol='0.01', max_iter=100)
    with pytest.raises(ValueError):
        kmeans = KMeans(k=5, tol=0.01, max_iter='100')
    # invalid k value
    with pytest.raises(ValueError):
        kmeans = KMeans(k=0, tol=0.01, max_iter=100)

def test_KMeans_small():
    # test for functionality of KMeans class

    # first, test with default make_clusters() param values
    pts, clusts = make_clusters()
    kmeans = KMeans(k=4) # make_clusters() defaults to generating 3 clusters, see if we can further stratify data
    kmeans.fit(pts)
    k_clusts = kmeans.predict(pts)
    # compare original clusters to our kmeans clusters
    #plot_clusters(pts, clusts)
    #plot_clusters(pts, k_clusts)
    # assert number identified clusters and number of centroids are both = 4
    assert len(kmeans.get_centroids()) == 4 
    assert len(set(k_clusts)) == 4

def test_KMeans_larger():
    # test for functionality of KMeans class

    # increase number of points to see scalability
    pts, clusts = make_clusters(n=5000)
    kmeans = KMeans(k=4) # make_clusters() defaults to generating 3 clusters, see if we can further stratify data
    kmeans.fit(pts)
    k_clusts = kmeans.predict(pts)
    # compare original clusters to our kmeans clusters
    #plot_clusters(pts, clusts)
    #plot_clusters(pts, k_clusts)
    # assert number identified clusters and number of centroids are both = 4
    assert len(kmeans.get_centroids()) == 4 
    assert len(set(k_clusts)) == 4

def test_KMeans_moreFeatures():
    # test for functionality of KMeans class

    # increase number of features to see scalability
    pts, clusts = make_clusters(m=3)
    kmeans = KMeans(k=4) # make_clusters() defaults to generating 3 clusters, see if we can further stratify data
    kmeans.fit(pts)
    k_clusts = kmeans.predict(pts)
    # compare original clusters to our kmeans clusters
    #plot_clusters(pts, clusts)
    #plot_clusters(pts, k_clusts)
    # assert number identified clusters and number of centroids are both = 4
    assert len(kmeans.get_centroids()) == 4 
    assert len(set(k_clusts)) == 4
