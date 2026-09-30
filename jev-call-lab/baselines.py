"""Non-LLM baselines scored at the same forecast points as Jev.

B0 constant 0.5 | B1 furthest stage reached | B2 logistic regression on event counts
(pure Python, trained on every prefix of the training calls).
"""
import math
import random

import evaluate
from schema import EVENTS
from simulate import attempted_booking, decision_index, forecast_points

STAGE_RANK = {
    "greeting": 0, "identity": 1, "handoff": 1, "intent": 2, "discovery": 2, "info": 2, "availability": 3,
    "offer": 4, "details": 5, "confirm": 6, "booking": 7, "wrapup": 0, "complete": 0,
}
_EVENTS = sorted(EVENTS)


def stage_reached(prefix):
    return max(STAGE_RANK[e["stage"]] for e in prefix) / 7


def features(prefix):
    counts = {name: 0 for name in _EVENTS}
    for e in prefix:
        counts[e["event"]] += 1
    fails = sum(not e["ok"] for e in prefix)
    return [1.0] + [math.log1p(counts[n]) for n in _EVENTS] + [math.log1p(fails), stage_reached(prefix)]


class LogReg:
    def __init__(self, epochs=30, lr=0.05, l2=1e-3):
        self.epochs, self.lr, self.l2, self.w = epochs, lr, l2, None

    def fit(self, X, y):
        self.w = [0.0] * len(X[0])
        order = list(range(len(X)))
        rng = random.Random(0)
        for _ in range(self.epochs):
            rng.shuffle(order)
            for j in order:
                x, t = X[j], y[j]
                g = self.predict_one(x) - t
                self.w = [w - self.lr * (g * xi + self.l2 * w) for w, xi in zip(self.w, x)]
        return self

    def predict_one(self, x):
        z = sum(w * xi for w, xi in zip(self.w, x))
        return 1 / (1 + math.exp(-max(-30, min(30, z))))


def _prefixes(calls):
    for c in calls:
        ev = c["events"]
        for i in forecast_points(ev):
            yield c, i, ev[: i + 1]


def _as_results(calls, score):
    out = []
    for c in calls:
        ev = c["events"]
        out.append({
            "label_booked": c["label_booked"],
            "n_events": len(ev),
            "decision_index": decision_index(ev),
            "attempted_booking": attempted_booking(ev),
            "turns": [{"event_index": i, "p_book": score(ev[: i + 1])} for i in forecast_points(ev)],
        })
    return out


def cross_validate(calls, k=5):
    """AUCs from k-fold out-of-fold predictions (for a set with no separate train split)."""
    scored = []
    for f in range(k):
        train = [c for i, c in enumerate(calls) if i % k != f]
        test = [c for i, c in enumerate(calls) if i % k == f]
        scored += [(name, r) for name, rs in _score(train, test).items() for r in rs]
    return {name: evaluate.auc_table([r for n, r in scored if n == name]) for name in dict(scored)}


def _score(train_calls, test_calls):
    rows = list(_prefixes(train_calls))
    model = LogReg().fit([features(p) for _, _, p in rows], [float(c["label_booked"]) for c, _, _ in rows])
    scorers = {
        "B0_constant": lambda p: 0.5,
        "B1_stage_reached": stage_reached,
        "B2_logreg_counts": lambda p: model.predict_one(features(p)),
    }
    return {name: _as_results(test_calls, score) for name, score in scorers.items()}


def evaluate_all(train_calls, test_calls):
    out = {}
    for name, res in _score(train_calls, test_calls).items():
        out[name] = {
            "auc": evaluate.auc_table(res),
            "brier": {k: evaluate._round(evaluate.brier(evaluate._rows(res, k))) for k in evaluate.POINTS},
        }
    return out
