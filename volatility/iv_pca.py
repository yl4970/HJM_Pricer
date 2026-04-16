import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt, matplotlib.ticker as mtick

import logging
from utils.logging import setup_logger
from utils.util import ANNUALIZE_FACTOR, MIN_PCA_K, MAX_PCA_K

logger = setup_logger(__name__)

def iv_pca(
        dF: np.ndarray,
        freq: str = 'D',
        min_k: int = MIN_PCA_K,
        max_k: int = MAX_PCA_K,
        elbow_graph: bool = False
):
    # First PCA pass: diagnostics only
    pca_full = PCA(svd_solver="covariance_eigh")
    pca_full.fit(dF)
    eigval_daily = pca_full.explained_variance_

    k, diag = choose_n_components(
        eigval = eigval_daily, 
        min_k = min_k,
        max_k = max_k,
        elbow_graph = elbow_graph
        )

    logger.verbose(
        "PCA diagnostics: chosen_k=%d, k_gap=%d, k_curv=%d",
        k, diag["k_gap"], diag["k_curvature"]
    )

    # Second PCA pass: actual truncation
    pca = PCA(
        n_components=k,
        svd_solver="covariance_eigh"
    )
    pca.fit(dF)

    logger.info(
        "PCA explained_variance_ratio (first %d): %s", 
        k,
        pca.explained_variance_ratio_
    )

    eigval_annual = pca.explained_variance_ * ANNUALIZE_FACTOR[freq]
    eigvec = pca.components_

    vols_annual = eigvec.T * np.sqrt(eigval_annual)

    return vols_annual
    
def choose_n_components(
        eigval: np.ndarray,
        min_k: int = MIN_PCA_K,
        max_k: int = MAX_PCA_K,
        elbow_graph: bool = False
):
    diag = pca_diagnostics(eigval)
    k = min_k

    if max(diag.values()) > k:
        logger.warning(
            "Unusual PCA spectrum detected,",
            extra=diag
        )
    
    if elbow_graph:
        k_to_graph = min(len(eigval), max_k)
        eigval_to_graph = eigval[:k_to_graph]
        explained_variance_ratio = np.array([i/sum(eigval) for i in eigval_to_graph])
        graph_elbow(k_to_graph, explained_variance_ratio)

    return k, diag

def graph_elbow(
        k_to_graph: int,
        explained_variance_ratio: np.ndarray
):
    x_range = range(1, k_to_graph+1) 
    y_range = range(0,110,10)
    plt.xticks(x_range)
    plt.plot(x_range, explained_variance_ratio*100, 'ro-')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.yticks(y_range)
    plt.grid()
    plt.rcParams["figure.figsize"] = (3,3)
        

def pca_diagnostics(eigval: np.ndarray):
    # minimum 3 n_components for rates
    n = 3

    # check where the largest delta of variance happens
    gaps = eigval[:-1] - eigval[1:]
    k_gap = gaps.argmax() + 1

    # check Elbow per curvature
    second_deriv = np.diff(eigval, n=2)
    k_curv = second_deriv.argmin() + 2

    return {
        "k_gap": k_gap,
        "k_curvature": k_curv,
    }