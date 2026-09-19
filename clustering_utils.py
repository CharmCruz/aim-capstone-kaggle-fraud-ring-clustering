"""
Reusable helper functions extracted from the analysis notebooks
(notebooks/Part_2_Step_3_5.ipynb), pulled out here so they can be
imported and reused independently of the notebook environment.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def hopkins_statistic(X, m=None, random_state=42):
    """
    Compute the Hopkins statistic for a feature matrix X, to test whether
    the data has meaningful cluster structure before running any clustering
    algorithm on it.

    H close to 1.0  -> strong cluster structure (data is clusterable)
    H close to 0.5  -> random/uniform data (no real cluster structure)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Scaled/preprocessed feature matrix.
    m : int, optional
        Number of sample points to draw. Defaults to 5% of n_samples.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    float
        The Hopkins statistic.
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    m = m or int(0.05 * n)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    u_dist, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    u_dist = u_dist[:, 1]
    rand_pts = rng.uniform(X.min(axis=0), X.max(axis=0), (m, X.shape[1]))
    w_dist, _ = nbrs.kneighbors(rand_pts, n_neighbors=1)
    return w_dist[:, 0].sum() / (u_dist.sum() + w_dist[:, 0].sum())


def hopkins_robustness_check(X, seeds=(7, 21, 100, 2024, 55555), sample_size=5000):
    """
    Re-run the Hopkins statistic across multiple random samples/seeds to
    confirm a result is stable rather than a sampling fluke.

    Returns
    -------
    dict
        {seed: hopkins_value, ...} plus 'mean' and 'std' keys.
    """
    results = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
        results[seed] = hopkins_statistic(X[idx], random_state=seed)
    vals = list(results.values())
    results["mean"] = float(np.mean(vals))
    results["std"] = float(np.std(vals))
    return results


def suggest_dbscan_eps(X_cluster, min_samples=10):
    """
    Auto-detect a reasonable DBSCAN `eps` value using the k-distance /
    "knee point" method: sort each point's distance to its min_samples-th
    nearest neighbor, then find the point of maximum curvature (the knee)
    on that curve.

    Parameters
    ----------
    X_cluster : array-like, shape (n_samples, n_features)
        Feature matrix to fit neighbors on.
    min_samples : int
        The DBSCAN min_samples value (also used as k for k-distance).

    Returns
    -------
    tuple(float, np.ndarray)
        (suggested_eps, sorted_k_distances) — the k_distances array is
        also returned so the caller can plot the k-distance curve.
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_cluster)
    distances, _ = neighbors.kneighbors(X_cluster)
    k_distances = np.sort(distances[:, -1])

    x = np.arange(len(k_distances))
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min())

    dist_from_line = np.abs(y_norm - x_norm) / np.sqrt(2)
    knee_idx = np.argmax(dist_from_line)
    eps_suggestion = k_distances[knee_idx]

    return float(eps_suggestion), k_distances


def fraud_concentration_by_cluster(cluster_labels, y_fraud, baseline_rate=None):
    """
    Given cluster assignments and the (validation-only) fraud label, compute
    each cluster's fraud rate, size, and concentration multiple vs. baseline.
    This is the core "so what" check used throughout this project: cluster
    quality metrics (Silhouette, Davies-Bouldin) describe separation, not
    business usefulness — this function answers the business question directly.

    Parameters
    ----------
    cluster_labels : array-like
        Cluster assignment per row (e.g. from KMeans.labels_, DBSCAN.labels_,
        or scipy's fcluster output).
    y_fraud : array-like
        The isFraud ground-truth label — used only for validation, never for
        fitting the clustering model itself.
    baseline_rate : float, optional
        Overall dataset fraud rate to compare against. If not given, it is
        computed from y_fraud directly.

    Returns
    -------
    pandas.DataFrame
        One row per cluster, with columns: cluster, n_transactions,
        fraud_rate_pct, concentration_x (fraud_rate / baseline_rate).
    """
    import pandas as pd

    df = pd.DataFrame({"cluster": cluster_labels, "isFraud": y_fraud})
    if baseline_rate is None:
        baseline_rate = df["isFraud"].mean()

    summary = (
        df.groupby("cluster")["isFraud"]
        .agg(n_transactions="count", fraud_rate=lambda s: s.mean() * 100)
        .reset_index()
    )
    summary["concentration_x"] = summary["fraud_rate"] / (baseline_rate * 100)
    return summary.sort_values("fraud_rate", ascending=False).reset_index(drop=True)
