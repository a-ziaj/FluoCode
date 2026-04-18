import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

log = logging.getLogger(__name__)


def _seq_identity(a, b):
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else 0.0


def _pairwise_identity(sequences):
    n   = len(sequences)
    mat = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _seq_identity(sequences[i], sequences[j])
            mat[i, j] = mat[j, i] = s
    return mat


def _assign_clusters_to_folds(cluster_labels, n_folds):
    """Greedy bin-packing: largest clusters first, assign to least-full fold."""
    counts = [(c, (cluster_labels == c).sum())
              for c in np.unique(cluster_labels)]
    counts.sort(key=lambda x: -x[1])

    fold_sizes  = np.zeros(n_folds, dtype=int)
    fold_assign = {}
    for cluster_id, size in counts:
        fold = int(np.argmin(fold_sizes))
        fold_assign[cluster_id] = fold
        fold_sizes[fold] += size

    return np.array([fold_assign[c] for c in cluster_labels])


def cluster_sequences(sequences: List[str],
                      identity_threshold: float = 0.3,
                      n_folds: int = 5) -> np.ndarray:
    """
    Cluster protein sequences by identity and assign fold labels.

    Random k-fold CV leaks information when similar proteins appear in both
    train and test. This keeps entire sequence clusters in the same fold so
    the test set is genuinely out-of-distribution at each split.

    Returns an int array [N] of fold assignments (0 to n_folds-1).
    """
    n = len(sequences)
    log.info(f"clustering {n} sequences (threshold={identity_threshold}, folds={n_folds})")

    dist_mat   = 1.0 - _pairwise_identity(sequences)
    n_clusters = min(n_folds * 4, n)
    clusterer  = AgglomerativeClustering(n_clusters=n_clusters,
                                         metric="precomputed",
                                         linkage="average")
    labels = clusterer.fit_predict(dist_mat)
    log.info(f"  {n_clusters} clusters formed")

    fold_labels = _assign_clusters_to_folds(labels, n_folds)
    for f in range(n_folds):
        log.info(f"  fold {f}: {(fold_labels == f).sum()} proteins")
    return fold_labels


def cluster_from_cdhit(clstr_path: str,
                       protein_names: List[str],
                       n_folds: int = 5):
    """
    Build fold labels from a CD-HIT .clstr file.
    Useful for large datasets where the O(N²) pairwise matrix is too slow.
    Returns None if the file doesn't exist.
    """
    from pathlib import Path
    if not Path(clstr_path).exists():
        log.warning(f"CD-HIT file not found: {clstr_path}")
        return None

    name_to_cluster, current = {}, -1
    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current += 1
            elif ">" in line:
                name = line.split(">")[1].split("...")[0].strip()
                name_to_cluster[name.lower()] = current

    log.info(f"CD-HIT: {current+1} clusters from {clstr_path}")

    name_idx = {n.lower(): i for i, n in enumerate(protein_names)}
    labels   = np.zeros(len(protein_names), dtype=int)
    missed   = 0
    for name, idx in name_idx.items():
        if name in name_to_cluster:
            labels[idx] = name_to_cluster[name]
        else:
            missed += 1
    if missed:
        log.warning(f"  {missed} proteins not in CD-HIT output (assigned cluster 0)")

    return _assign_clusters_to_folds(labels, n_folds)


def get_cv_splits(fold_labels: np.ndarray,
                  n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    all_idx = np.arange(len(fold_labels))
    return [(all_idx[fold_labels != f], all_idx[fold_labels == f])
            for f in range(n_folds)]
