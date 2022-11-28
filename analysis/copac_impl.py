"""COPAC implementation (COrrelation PArtition Clustering)

which aims at improved robustness, completeness, usability, and efficiency.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def covariance_matrix(m):
    """Computes the Covariance matrix"""

    length = len(m)
    cols = m.shape[1]
    mean = np.mean(m, axis=0)
    cov = np.empty([cols, cols])
    
    for i in range(cols):
        for j in range(cols):
            cov[i][j] = np.dot((m[:, i] - mean[i]), (m[:, j] - mean[j])) / length
            
    return cov


def loc_corr_dim(eigenvals, alpha = 0.85):
    """The local correlation dimensionality"""

    all_eigs = sum(eigenvals)
    corr_dim = 0

    for i in range(1, len(eigenvals)):
        ratio = sum(eigenvals[:i]) / all_eigs
        if ratio >= alpha:
            corr_dim = i
            break

    if corr_dim == 0:
        corr_dim = 1

    return corr_dim


def corr_dist_matrix(p, corr_dim, eigenvals, eigenvecs):
    """Correlation distance matrix"""

    length = len(eigenvals)
    adapted_eig_mat = np.zeros([length, length])
    
    for i in range(length):
        if eigenvals[i] > corr_dim:
            adapted_eig_mat[i][i] = 1

    return eigenvecs @ adapted_eig_mat @ eigenvecs.T


def corr_dist_measure(p, q, m):
    """The correlation distance measure"""

    return np.sqrt((p - q) @ m @ (p - q).T)


def copac(X, k=2, mu=5, eps=0.5, alpha=0.85):
    """Performs correlation clustering using COPAC algorithm.

    Parameters
    ----------
    X: numpy.array
        The feature array.
    k: int, default=2
        The number of points considered to compute the neighborhood of a point P. It turned out that for k=3, d was robust in all our tests. In general, setting 3 · d ≤ k seems to be a reasonable suggestion.
    mu: int, default=5
        The parameter µ specifies the minimum number of points in a cluster and, therefore, is quite intuitive. Obviously, µ ≤ k should hold.
    eps: float, default=0.5
        The parameter ε is used to specify the neighborhood predicate and can be choosen as proposed in [6] and [13].
    alpha: float, default=0.85
        Computes the correlation dimensionality λP of a point P ∈ D. As discussed above, this parameter is very robust in the range between 0.8 ≤ α ≤ 0.9. Thus, we choose α = 0.85 throughout all our experiments.

    Returns
    -------
    labels: numpy.array
        The labeled cluster array.
    """

    n, d = X.shape
    y = np.empty(n)
    lambda_ = np.empty(0)
    M_hat = list()
        
        
    knn = NearestNeighbors(n_neighbors=k)
    
    knn.fit(X)
    knns = knn.kneighbors(return_distance=False)
    
    for P, nn in enumerate(knns):
        N_P = X[nn]
        
        sig = np.cov(N_P[:, :], rowvar=False, ddof=0)

        vals, eigenvecs = np.linalg.eigh(sig)
        eigenvals = np.sort(vals)[::-1]
        
        lambda_P = loc_corr_dim(eigenvals, alpha=alpha)
        lambda_ = np.append(lambda_, int(lambda_P))
        
        M_hat.append(corr_dist_matrix(X[P], lambda_P, eigenvals, eigenvecs))

    sorted_indeces = np.argsort(lambda_)
    sorted_lambda = lambda_[sorted_indeces]
    
    counter = 1
    Ds = list()
    d_lambda = list()

    for i in range(len(sorted_lambda)):

        if sorted_lambda[i] == counter:
            d_lambda.append(sorted_indeces[i])
        else:
            Ds.append(np.array(d_lambda))
            counter += 1
            d_lambda.clear()
        
            if sorted_lambda[i] == counter:
                d_lambda.append(sorted_indeces[i])
                
            else:
                while sorted_lambda[i] != counter:
                    Ds.append(np.array(d_lambda))
                    counter += 1
                else:
                    d_lambda.append(sorted_indeces[i])
                    Ds.append(np.array(d_lambda))
    
    Ds.append(np.array(d_lambda))
     
    clusterlabels = np.empty(0)
    numberoflabels = 0
    
    for D in Ds:
        length = len(D)
        if length >= 1:
            cdistance_matrix = np.zeros((length, length))
            for i in range(length):
                for j in range(i, length):
                    p = X[D[i]]
                    q = X[D[j]]
                    dist = max([corr_dist_measure(p, q, M_hat[D[i]]), corr_dist_measure(q, p, M_hat[D[j]])])
                    cdistance_matrix[i][j] = dist
                    cdistance_matrix[j][i] = dist

            cdistance_matrix = np.sqrt(cdistance_matrix)
            cluster = DBSCAN(eps=eps, min_samples=mu, metric='precomputed', n_jobs=-1)
            cluster.fit(cdistance_matrix)
            
            for g in range(len(cluster.labels_)):
                if cluster.labels_[g] >= 0:
                    clusterlabels = np.append(clusterlabels, cluster.labels_)
                    
            numberoflabels += np.unique(clusterlabels).size
            new_labels = cluster.labels_.copy()
            
            for h in range(len(new_labels)):
                if new_labels[h] >= 0:
                    new_labels[h] += numberoflabels            
            
            for k in range(length):
                y[D[k]] = new_labels[k]
            
        else:
            pass
        
    return y

