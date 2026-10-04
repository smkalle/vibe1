"""On-disk project workspace (spec v2 §10): samples, runs, manifests, decisions.

    workspace/<project>/project.json
    workspace/<project>/samples/<sample_id>.json
    workspace/<project>/runs/<run_id>/manifest.json   (never the key)
    workspace/<project>/runs/<run_id>/results.jsonl   (one line per call, appended -> resumable)
    workspace/<project>/runs/<run_id>/errors.jsonl

The root is ./workspace (gitignored) or $JEV_WORKSPACE.
"""
import hashlib
import json
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

# Built-in corpora. B2 trains on `train` minus the sample (held out), never on the sample.
BUILTIN_CORPORA = {
    "synthetic": {"label": "Synthetic phone calls", "file": "synthetic_calls.json", "train": "synthetic_calls.json",
                  "about": "Seeded phone-style calls: silence, hangups, identity failures, recoveries."},
    "sgd_test": {"label": "SGD test (real dialogues)", "file": "sgd_test.json", "train": "sgd_train.json",
                 "about": "Google SGD salon/therapist appointment dialogues, reduced to structure."},
    "sgd_dev": {"label": "SGD dev (real dialogues)", "file": "sgd_dev.json", "train": "sgd_train.json",
                "about": "Small SGD split (44 calls)."},
    "sgd_train": {"label": "SGD train (real dialogues)", "file": "sgd_train.json", "train": "sgd_train.json",
                  "about": "Large SGD split; rules then train on the calls outside the sample."},
}

# Manifest fields allowed on disk. Anything else (the key above all) is dropped.
MANIFEST_FIELDS = {
    "run_id", "project", "corpus_id", "corpus_hash", "sample_id", "approach", "scorer", "mode", "model", "url",
    "effort", "questions_hash", "prompt_hash", "concurrency", "git_sha", "started", "finished", "status",
    "n_calls", "n_done", "n_failed", "cost_usd", "cap_usd", "probe", "source", "note",
}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_lock = threading.Lock()


def root() -> Path:
    return Path(os.environ.get("JEV_WORKSPACE", HERE / "workspace"))


def load_corpus(corpus_id: str) -> list:
    return json.loads((DATA / BUILTIN_CORPORA[corpus_id]["file"]).read_text())


def load_train(corpus_id: str) -> list:
    return json.loads((DATA / BUILTIN_CORPORA[corpus_id]["train"]).read_text())


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def list_projects(base: Path | None = None) -> list[str]:
    r = base or root()
    return sorted(p.name for p in r.iterdir() if (p / "project.json").exists()) if r.exists() else []


class Project:
    def __init__(self, name: str, base: Path | None = None):
        self.name = _SAFE.sub("-", name).strip("-") or "default"
        self.dir = (base or root()) / self.name

    # -- project ---------------------------------------------------------------
    def ensure(self) -> "Project":
        for sub in ("samples", "runs"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        if not (self.dir / "project.json").exists():
            self._write_json(self.dir / "project.json", {"name": self.name, "created": now(), "decisions": []})
        return self

    def meta(self) -> dict:
        return json.loads((self.dir / "project.json").read_text())

    def record_decision(self, use_case: str, approach: str, note: str, evidence: dict):
        meta = self.meta()
        meta["decisions"] = [d for d in meta["decisions"] if d["use_case"] != use_case] + [
            {"use_case": use_case, "approach": approach, "note": note, "evidence": evidence, "at": now()}]
        self._write_json(self.dir / "project.json", meta)

    # -- samples ---------------------------------------------------------------
    def save_sample(self, sample: dict) -> dict:
        self._write_json(self.dir / "samples" / f"{sample['sample_id']}.json", sample)
        return sample

    def samples(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.dir / "samples").glob("*.json"))]

    def sample(self, sample_id: str) -> dict:
        return json.loads((self.dir / "samples" / f"{sample_id}.json").read_text())

    # -- runs ------------------------------------------------------------------
    def new_run(self, **fields) -> str:
        run_id = _SAFE.sub("-", f"{fields['approach']}-{fields['mode']}-{fields['sample_id']}-"
                                f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
        (self.dir / "runs" / run_id).mkdir(parents=True)
        self.write_manifest(run_id, {"run_id": run_id, "project": self.name, "started": now(), "status": "running",
                                     "n_done": 0, "n_failed": 0, "cost_usd": 0.0, "git_sha": git_sha(), **fields})
        return run_id

    def write_manifest(self, run_id: str, manifest: dict):
        clean = {k: v for k, v in manifest.items() if k in MANIFEST_FIELDS}
        self._write_json(self.dir / "runs" / run_id / "manifest.json", clean)

    def update_manifest(self, run_id: str, **fields):
        m = self.manifest(run_id)
        m.update(fields)
        self.write_manifest(run_id, m)

    def manifest(self, run_id: str) -> dict:
        return json.loads((self.dir / "runs" / run_id / "manifest.json").read_text())

    def append(self, run_id: str, kind: str, row: dict):
        with _lock, open(self.dir / "runs" / run_id / f"{kind}.jsonl", "a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def results(self, run_id: str, kind: str = "results") -> list[dict]:
        p = self.dir / "runs" / run_id / f"{kind}.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        seen, out = set(), []
        for r in reversed(rows):  # last write wins if a call was re-scored
            if r["call_id"] not in seen:
                seen.add(r["call_id"])
                out.append(r)
        return list(reversed(out))

    def runs(self, sample_id: str | None = None) -> list[dict]:
        out = []
        for d in sorted((self.dir / "runs").glob("*/manifest.json")):
            m = json.loads(d.read_text())
            if sample_id is None or m.get("sample_id") == sample_id:
                out.append(m)
        return sorted(out, key=lambda m: m.get("started", ""))

    def latest_run(self, sample_id: str, approach: str, complete_only: bool = False) -> dict | None:
        cands = [m for m in self.runs(sample_id) if m["approach"] == approach
                 and (not complete_only or m["status"] == "done")]
        return cands[-1] if cands else None

    def done_ids(self, run_id: str) -> set:
        return {r["call_id"] for r in self.results(run_id)}

    def import_results(self, sample: dict, results: list, approach: str, mode: str, model: str,
                       questions_hash: str, note: str = "") -> str:
        """Attach results produced elsewhere to a sample. Provenance is stated by the operator, never guessed.

        Raises ValueError unless the results cover the sample (>= 95% of its calls, nothing outside it).
        """
        ids = {r["call_id"] for r in results if "turns" in r}
        sample_ids = set(sample["call_ids"])
        outside = ids - sample_ids
        if outside:
            raise ValueError(f"{len(outside)} result calls are not in sample {sample['sample_id']} "
                             f"(e.g. {sorted(outside)[0]}); freeze a sample from the same corpus first")
        if len(ids) < 0.95 * len(sample_ids):
            raise ValueError(f"results cover {len(ids)}/{len(sample_ids)} sample calls; need at least 95%")
        run_id = self.new_run(approach=approach, scorer=approach, mode=mode, model=model, sample_id=sample["sample_id"],
                              corpus_id=sample["corpus_id"], corpus_hash=sample["corpus_hash"],
                              questions_hash=questions_hash, n_calls=len(sample_ids), source="imported", note=note)
        for r in results:
            if "turns" in r:
                self.append(run_id, "results", r)
        self.update_manifest(run_id, status="done", finished=now(), n_done=len(ids),
                             cost_usd=_run_cost(results))
        return run_id

    @staticmethod
    def _write_json(path: Path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1))
        tmp.replace(path)


def _run_cost(results) -> float:
    return sum(t.get("cost_usd", 0) for r in results for t in r.get("turns", [])) + \
        sum(r.get("review_cost_usd", 0) for r in results)
