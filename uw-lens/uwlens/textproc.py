"""Text utilities: bilingual tokenization, language detection, chunking and PHI redaction."""
from __future__ import annotations

import re
import unicodedata

# Hiragana, Katakana, CJK unified ideographs (+ext A), half-width katakana.
_CJK = r"぀-ヿ㐀-䶿一-鿿ｦ-ﾟ"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_LATIN_WORD = re.compile(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*")
_PIECE_END = re.compile(r"(?<=[.!?。！？\n])")

_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with "
    "what which who how when where does do can".split()
)


def normalize(text: str) -> str:
    # NFKC folds full-width latin/digits (ＨｂＡ１ｃ -> HbA1c) common in Japanese documents.
    return unicodedata.normalize("NFKC", text)


def tokenize(text: str) -> list[str]:
    """Latin words (lower-cased, stop-words dropped) plus CJK character bigrams.

    Japanese has no spaces, so overlapping bigrams are the standard dependency-free
    approach for keyword retrieval; single-character runs are kept as unigrams.
    """
    text = normalize(text).lower()
    tokens = [w for w in _LATIN_WORD.findall(text) if w not in _STOP]
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def detect_language(text: str) -> str:
    """Return 'ja', 'en' or 'mixed' from the share of CJK characters among letters."""
    cjk = len(re.findall(f"[{_CJK}]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = cjk + latin
    if total == 0:
        return "en"
    share = cjk / total
    if share >= 0.6:
        return "ja"
    if share >= 0.15:
        return "mixed"
    return "en"


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Sentence/line-aware sliding window that preserves the original whitespace.

    Pieces end after a sentence terminator or newline; consecutive pieces are packed up to
    `max_chars`, and each new chunk starts with up to `overlap` trailing chars of the previous one.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    overlap = min(overlap, max_chars // 2)
    pieces = [p for p in _PIECE_END.split(text) if p]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def flush():
        joined = "".join(current).strip()
        if joined:
            chunks.append(joined)

    for piece in pieces:
        while len(piece) > max_chars:  # a single over-long sentence/line: hard split
            flush()
            current, size = [], 0
            chunks.append(piece[:max_chars].strip())
            piece = piece[max_chars - overlap :]
        if size + len(piece) > max_chars:
            flush()
            tail: list[str] = []  # carry whole trailing pieces (<= overlap chars) into the next chunk
            for prev in reversed(current):
                if sum(map(len, tail)) + len(prev) > overlap:
                    break
                tail.insert(0, prev)
            current = tail if sum(map(len, tail)) + len(piece) <= max_chars else []
            size = sum(map(len, current))
        current.append(piece)
        size += len(piece)
    flush()
    return chunks


# --- PHI redaction -----------------------------------------------------------------
# Order matters: specific identifiers before generic digit runs (phones).
_PHI_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MY_NUMBER", re.compile(r"(?<!\d)\d{4}[ \-]?\d{4}[ \-]?\d{4}(?!\d)")),
    (
        "MEMBER_ID",
        re.compile(
            r"(?i)(?:member|policy|subscriber|patient|mrn|insured|application|applicant)\s*(?:id|no\.?|number|#)\s*[:：#]?\s*"
            r"[A-Z0-9][A-Z0-9\-]{3,}"
            r"|(?:被保険者番号|保険者番号|記号番号|証番号|患者番号)\s*[:：]?\s*[0-9A-Za-z０-９\-－・]+"
        ),
    ),
    (
        "NAME",
        re.compile(r"(?im)(?:(?:patient|applicant|insured|member)\s+)?name\s*[:：][ \t]*[^\n,;·|]{2,60}"
                   r"|(?:氏名|患者名|被保険者氏名)\s*[:：][ \t　]*[^\s,，、。]{1,20}(?:[ 　][^\s,，、。]{1,20})?"),
    ),
    (
        "DOB",
        re.compile(
            r"(?i)(?:dob|date of birth|birth ?date|生年月日)\s*[:：]?\s*"
            r"(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
            r"(?:昭和|平成|令和)\s*\d{1,2}年\s*\d{1,2}月\s*\d{1,2}日)"
        ),
    ),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?81[ \-]?|\+?1[ \-]?)?\(?0?\d{2,4}\)?[ \-]\d{2,4}[ \-]\d{3,4}(?![\w-])")),
]


def redact_phi(text: str) -> tuple[str, dict[str, int]]:
    """Mask common identifiers. Returns (redacted_text, counts_by_kind).

    Pattern-based redaction is a safety net for demo/research corpora, not a
    de-identification guarantee (HIPAA Safe Harbor / APPI need a reviewed process).
    """
    text = normalize(text)
    counts: dict[str, int] = {}
    for kind, pattern in _PHI_PATTERNS:
        text, n = pattern.subn(f"[{kind}]", text)
        if n:
            counts[kind] = counts.get(kind, 0) + n
    return text, counts


def find_phi(text: str) -> list[str]:
    """Return the PHI kinds still detectable in `text` (used by tests and the indexer audit)."""
    text = normalize(text)
    return [kind for kind, pattern in _PHI_PATTERNS if pattern.search(text)]
