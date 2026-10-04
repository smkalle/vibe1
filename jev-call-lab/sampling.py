"""Frozen, stratified, seeded benchmark samples (spec v2 §5 ③, eval W3).

Every approach in a benchmark scores exactly one frozen sample, so comparisons are paired.
Samples are never "the first N calls": corpus files are ordered by service, and the first
calls of sgd_test are all unbooked therapist calls.
"""
import hashlib
import json
import math
import random
from collections import defaultdict

STRATA = ("label_booked", "source", "specialty")


def corpus_hash(calls: list) -> str:
    canon = json.dumps(calls, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _stratum(call: dict, strata) -> tuple:
    return tuple(call.get(k) for k in strata)


def stratified_ids(calls: list, n: int, seed: int, strata=STRATA) -> list[str]:
    """n call ids, allocated to strata in proportion (largest remainder), shuffled within strata.

    When both outcomes exist in the corpus and n >= 2, the sample always holds both.
    """
    n = max(0, min(n, len(calls)))
    groups = defaultdict(list)
    for c in calls:
        groups[_stratum(c, strata)].append(c["call_id"])
    rng = random.Random(seed)
    for key in sorted(groups, key=repr):
        groups[key].sort()
        rng.shuffle(groups[key])

    total = len(calls)
    quota = {k: n * len(v) / total for k, v in groups.items()}
    alloc = {k: math.floor(q) for k, q in quota.items()}
    rest = n - sum(alloc.values())
    for k in sorted(groups, key=lambda k: (-(quota[k] - alloc[k]), repr(k)))[:rest]:
        alloc[k] += 1

    # Guarantee both label classes (only label_booked is guaranteed; it is what AUC needs).
    if n >= 2 and "label_booked" in strata:
        li = strata.index("label_booked")
        for label in (True, False):
            have = sum(a for k, a in alloc.items() if k[li] is label)
            avail = [k for k in groups if k[li] is label and alloc[k] < len(groups[k])]
            if have == 0 and avail:
                donor = max((k for k in alloc if alloc[k] > 0 and k[li] is not label), key=lambda k: (alloc[k], repr(k)))
                alloc[donor] -= 1
                alloc[max(avail, key=lambda k: (len(groups[k]), repr(k)))] += 1

    ids = [cid for k in sorted(groups, key=repr) for cid in groups[k][: alloc[k]]]
    return sorted(ids)


def freeze(corpus_id: str, calls: list, n: int, seed: int = 7, strata=STRATA) -> dict:
    ids = stratified_ids(calls, n, seed, strata)
    chash = corpus_hash(calls)
    sid = f"{corpus_id}-n{len(ids)}-s{seed}-" + hashlib.sha256(",".join(ids).encode()).hexdigest()[:8]
    by_id = {c["call_id"]: c for c in calls}
    return {
        "sample_id": sid,
        "corpus_id": corpus_id,
        "corpus_hash": chash,
        "seed": seed,
        "strata": list(strata),
        "n": len(ids),
        "n_booked": sum(by_id[i]["label_booked"] for i in ids),
        "call_ids": ids,
    }


def auc_halfwidth(n: int, booked_rate: float, auc: float = 0.8) -> float | None:
    """Approximate 95% CI half-width of an AUC on n calls (Hanley & McNeil 1982).

    Used only to size samples before spending money; reported CIs come from the bootstrap.
    """
    n_pos = round(n * booked_rate)
    n_neg = n - n_pos
    if n_pos < 1 or n_neg < 1:
        return None
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return 1.96 * math.sqrt(max(var, 0.0))
