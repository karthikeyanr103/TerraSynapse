"""
evaluation/metrics.py
FAISS-backed retrieval evaluation: F1@K, Recall@K, mAP@K, and average
per-query retrieval time, for any query/gallery direction.
"""

import time
import numpy as np
import faiss
from tqdm import tqdm

from config.config import EMBED_DIM, TOP_K


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Flat inner-product index = cosine similarity, since vectors are L2-normalized."""
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings.astype(np.float32))
    return index


def _precision_recall_f1_at_k(ret_labels, query_label, total_relevant, k):
    top_k = ret_labels[:k]
    n_rel = sum(1 for lbl in top_k if lbl == query_label)
    precision = n_rel / k
    recall = n_rel / max(total_relevant, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _average_precision_at_k(ret_labels, query_label, k):
    """Standard AP@k: average of precision@i for each relevant hit within top-k."""
    hits = 0
    precisions = []
    for i, lbl in enumerate(ret_labels[:k], start=1):
        if lbl == query_label:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_retrieval(query_embeddings, query_labels,
                        gallery_embeddings, gallery_labels,
                        direction: str = "SAR_to_MS"):
    """
    Full retrieval evaluation for one query->gallery direction.
    Returns a dict with F1@5, F1@10, Recall@5, Recall@10, mAP@5, mAP@10,
    and avg_time_ms (average wall-clock FAISS search time per query).
    """
    index = build_faiss_index(gallery_embeddings)
    gallery_labels = np.array(gallery_labels)

    metrics = {f"F1@{k}": [] for k in TOP_K}
    metrics.update({f"Recall@{k}": [] for k in TOP_K})
    metrics.update({f"mAP@{k}": [] for k in TOP_K})
    times = []

    for i in tqdm(range(len(query_embeddings)), desc=f"Evaluating {direction}", unit="query"):
        q = query_embeddings[i:i + 1]
        q_lbl = query_labels[i]
        total_relevant = int(np.sum(gallery_labels == q_lbl))

        t0 = time.perf_counter()
        _, idxs = index.search(q, max(TOP_K))
        times.append((time.perf_counter() - t0) * 1000)

        ret_labels = gallery_labels[idxs[0]]

        for k in TOP_K:
            _, recall, f1 = _precision_recall_f1_at_k(ret_labels, q_lbl, total_relevant, k)
            ap = _average_precision_at_k(ret_labels, q_lbl, k)
            metrics[f"F1@{k}"].append(f1)
            metrics[f"Recall@{k}"].append(recall)
            metrics[f"mAP@{k}"].append(ap)

    result = {"direction": direction, "n_queries": len(query_embeddings),
               "avg_time_ms": float(np.mean(times))}
    for key, vals in metrics.items():
        result[key] = float(np.mean(vals))

    return result


def print_results_table(results_list):
    print("=" * 70)
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    for res in results_list:
        print(f"\n  Direction: {res['direction']}  (n_queries={res['n_queries']})")
        for k in TOP_K:
            print(f"    F1@{k}={res[f'F1@{k}']:.4f}  "
                  f"Recall@{k}={res[f'Recall@{k}']:.4f}  "
                  f"mAP@{k}={res[f'mAP@{k}']:.4f}")
        print(f"    Avg retrieval time: {res['avg_time_ms']:.3f} ms/query")
    print("=" * 70)
