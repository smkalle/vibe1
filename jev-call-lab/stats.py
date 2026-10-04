"""Statistics for paired benchmarks (spec v2 §7, eval W4).

All resampling is a cluster bootstrap over CALLS (turns within a call are correlated),
and every approach is resampled with the same call indices, so differences are paired.
"""
import random


def auc(scores: list, labels: list, min_class: int = 1) -> float | None:
    """Rank-based Mann-Whitney AUC (ties count half). O(n log n)."""
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos < max(1, min_class) or n_neg < max(1, min_class):
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(scores, labels):
    return sum((p - y) ** 2 for p, y in zip(scores, labels)) / len(scores) if scores else None


def ece(scores, labels, bins: int = 10):
    """Expected calibration error: bin-size-weighted |mean p - observed rate|."""
    if not scores:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(scores) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            total += len(idx) * abs(sum(scores[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx))
    return total / len(scores)


def reliability(scores, labels, bins: int = 5):
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(scores) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx),
                        "mean_p": sum(scores[i] for i in idx) / len(idx),
                        "observed": sum(labels[i] for i in idx) / len(idx)})
    return out


def macro_f1(pred: list, ref: list) -> float | None:
    pairs = [(p, r) for p, r in zip(pred, ref) if p is not None and r is not None]
    if not pairs:
        return None
    classes = sorted({c for pr in pairs for c in pr})
    f1s = []
    for c in classes:
        tp = sum(p == c and r == c for p, r in pairs)
        fp = sum(p == c and r != c for p, r in pairs)
        fn = sum(p != c and r == c for p, r in pairs)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def precision_recall_at(scores, labels, threshold: float):
    """Escalation view: flag a call as 'will not book' when P(book) < threshold."""
    flagged = [not y for p, y in zip(scores, labels) if p < threshold]
    actual_neg = sum(1 for y in labels if not y)
    tp = sum(flagged)
    precision = tp / len(flagged) if flagged else None
    recall = tp / actual_neg if actual_neg else None
    return {"threshold": threshold, "flagged": len(flagged), "precision": precision, "recall": recall}


def percentile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def paired_bootstrap(metric_fns: dict, n_calls: int, b: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap every metric with SHARED call resamples.

    metric_fns: {name: fn(indices) -> float | None}. Returns
      {"point": {name: v}, "ci": {name: (lo, hi)}, "diff": {(a, b): (mean, lo, hi)}}
    Resamples where a metric is undefined (e.g. one class) are skipped for that metric/pair.
    """
    rng = random.Random(seed)
    full = list(range(n_calls))
    point = {k: f(full) for k, f in metric_fns.items()}
    draws = {k: [] for k in metric_fns}
    pair_draws = {(a, c): [] for a in metric_fns for c in metric_fns if a != c}
    for _ in range(b):
        idx = [rng.randrange(n_calls) for _ in full]
        vals = {k: f(idx) for k, f in metric_fns.items()}
        for k, v in vals.items():
            if v is not None:
                draws[k].append(v)
        for (a, c) in pair_draws:
            if vals[a] is not None and vals[c] is not None:
                pair_draws[(a, c)].append(vals[a] - vals[c])
    ci = {k: (percentile(v, alpha / 2), percentile(v, 1 - alpha / 2)) if len(v) >= 20 else (None, None)
          for k, v in draws.items()}
    diff = {}
    for pair, v in pair_draws.items():
        diff[pair] = ((sum(v) / len(v), percentile(v, alpha / 2), percentile(v, 1 - alpha / 2))
                      if len(v) >= 20 else (None, None, None))
    return {"point": point, "ci": ci, "diff": diff}
