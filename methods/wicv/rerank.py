#!/usr/bin/env python3
"""K-reciprocal re-ranking (Zhong et al., CVPR 2017) for retrieval evaluation.

Operates on the three pairwise distance blocks. Memory scales with
(num_query + num_gallery)^2; for the full benchmark (~19k images) expect
roughly 6-8 GB of RAM, per-condition subsets are much lighter.
"""

from __future__ import annotations

import numpy as np


def k_reciprocal_rerank(
    q_g_dist: np.ndarray,
    q_q_dist: np.ndarray,
    g_g_dist: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Return the re-ranked query-to-gallery distance matrix."""
    query_num = q_g_dist.shape[0]
    all_num = query_num + g_g_dist.shape[0]

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    original_dist = np.power(original_dist, 2)
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    initial_rank = np.argsort(original_dist).astype(np.int32)

    V = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(all_num):
        forward_k = initial_rank[i, : k1 + 1]
        backward_k = initial_rank[forward_k, : k1 + 1]
        fi = np.where(backward_k == i)[0]
        k_reciprocal_index = forward_k[fi]

        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward = initial_rank[candidate_forward, : int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_k_reciprocal = candidate_forward[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal, k_reciprocal_index)) > 2 / 3 * len(
                candidate_k_reciprocal
            ):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    inverted_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        indices_non_zero = np.where(V[i, :] != 0)[0]
        for j in indices_non_zero:
            temp_min[0, inverted_index[j]] += np.minimum(V[i, j], V[inverted_index[j], j])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:, query_num:]
