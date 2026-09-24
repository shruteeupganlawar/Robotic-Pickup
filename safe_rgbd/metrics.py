import numpy as np


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, tn, fp, fn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def mae(estimated, truth, mask=None):
    e = np.asarray(estimated, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    diff = np.abs(e - t)
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean()) if diff.size else float("nan")


def rmse(estimated, truth, mask=None):
    e = np.asarray(estimated, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    diff = (e - t) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(diff.mean())) if diff.size else float("nan")


def latency_stats(times_ms):
    t = np.sort(np.asarray(times_ms, dtype=np.float64))
    return {
        "n": int(t.size),
        "mean_ms": float(t.mean()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "max_ms": float(t.max()),
    }


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
