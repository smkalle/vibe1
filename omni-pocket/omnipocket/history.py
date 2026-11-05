"""Local, append-only JSONL logs: conversation turns and benchmark runs.
Nothing leaves the device. Media is not stored, only its shape."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict) -> dict:
        record = {"ts": round(time.time(), 3), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn last line after a crash
        return rows[-limit:] if limit else rows

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def logs(data_dir: Path) -> tuple[Log, Log]:
    data_dir = Path(data_dir)
    return Log(data_dir / "history.jsonl"), Log(data_dir / "bench.jsonl")


def summarize(rows: list[dict]) -> dict:
    """Group benchmark rows by (engine, model, mode) -> median latency stats."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("engine"), r.get("model"), r.get("mode", "")), []).append(r)
    out = {}
    for key, items in groups.items():
        def med(field):
            vals = [x[field] for x in items if isinstance(x.get(field), (int, float))]
            return round(statistics.median(vals), 3) if vals else None

        out[" | ".join(str(k) for k in key if k)] = {
            "runs": len(items),
            "ttft_s": med("ttft_s"),
            "total_s": med("total_s"),
            "tokens_per_s": med("tokens_per_s"),
        }
    return out
