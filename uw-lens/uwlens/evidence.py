"""Evidence pack: pinned search results + analyst notes, exported as Markdown or CSV."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone

DISCLAIMER = (
    "Research aid only. Retrieved passages are evidence for a licensed underwriter's review, "
    "not a medical opinion or an underwriting decision."
)


@dataclass
class PinnedItem:
    id: str
    path: str
    title: str
    modality: str
    source: str
    language: str
    locator: str
    snippet: str
    query: str
    score: float
    note: str = ""

    @classmethod
    def from_hit(cls, hit, query: str) -> "PinnedItem":
        r = hit.record
        return cls(r["id"], r["path"], r["title"], r["modality"], r["source"], r["language"], r.get("locator", ""),
                   r["text"][:600], query, round(float(hit.score), 4))


@dataclass
class EvidencePack:
    case_ref: str = ""
    analyst: str = ""
    summary: str = ""
    items: list[PinnedItem] = field(default_factory=list)

    def pin(self, item: PinnedItem) -> bool:
        if any(i.id == item.id for i in self.items):
            return False
        self.items.append(item)
        return True

    def unpin(self, item_id: str) -> None:
        self.items = [i for i in self.items if i.id != item_id]

    def to_markdown(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        lines = [
            f"# Underwriting evidence pack — {self.case_ref or 'unreferenced case'}",
            "",
            f"- Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
            f"- Analyst: {self.analyst or '—'}",
            f"- Items: {len(self.items)}",
            "",
            f"> {DISCLAIMER}",
            "",
        ]
        if self.summary.strip():
            lines += ["## Analyst summary", "", self.summary.strip(), ""]
        lines += ["## Evidence", ""]
        for n, it in enumerate(self.items, 1):
            loc = f" · {it.locator}" if it.locator else ""
            lines += [
                f"### {n}. {it.title}",
                "",
                f"- File: `{it.path}`{loc}",
                f"- Source: {it.source} · Modality: {it.modality} · Language: {it.language}",
                f"- Found by query: “{it.query}” (score {it.score})",
            ]
            if it.note.strip():
                lines.append(f"- Note: {it.note.strip()}")
            quote = it.snippet.strip().replace("\n", "\n> ")
            lines += ["", f"> {quote}", ""]
        return "\n".join(lines)

    def to_csv(self) -> str:
        buf = io.StringIO()
        fields = ["n", "case_ref", "title", "path", "locator", "source", "modality", "language", "query", "score",
                  "note", "snippet"]
        w = csv.DictWriter(buf, fieldnames=fields)
        w.writeheader()
        for n, it in enumerate(self.items, 1):
            w.writerow({"n": n, "case_ref": self.case_ref, "title": it.title, "path": it.path, "locator": it.locator,
                        "source": it.source, "modality": it.modality, "language": it.language, "query": it.query,
                        "score": it.score, "note": it.note, "snippet": it.snippet})
        return buf.getvalue()
