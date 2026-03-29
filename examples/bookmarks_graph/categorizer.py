"""TF-IDF based cluster labeling and category persistence."""

import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

CATEGORIES_PATH = Path.home() / ".bookmarks_graph_categories.json"

# Common words to exclude from labels
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it",
    "its", "i", "you", "we", "they", "he", "she", "my", "your", "our",
    "their", "from", "by", "as", "up", "about", "into", "through",
    "just", "like", "so", "get", "can", "also", "more", "all", "if",
    "what", "when", "how", "new", "one", "not", "any", "https", "t",
    "co", "http", "amp", "rt", "via", "re", "im", "its", "ive",
    "dont", "doesnt", "cant", "wont", "isnt",
}


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"http\S+", "", text)          # strip URLs
    text = re.sub(r"@\w+", "", text)             # strip mentions
    text = re.sub(r"#(\w+)", r"\1", text)        # keep hashtag text
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in _STOPWORDS and len(w) > 2]


def label_clusters(
    partition: Dict[str, int],
    tweets: Dict[str, str],  # {tweet_id: text}
    top_n: int = 5,
) -> Dict[int, str]:
    """Return {community_id: label} using TF-IDF scoring across clusters."""
    # Group texts by community
    clusters: Dict[int, List[str]] = defaultdict(list)
    for tweet_id, community in partition.items():
        if tweet_id in tweets:
            clusters[community].append(tweets[tweet_id])

    total_docs = sum(len(v) for v in clusters.values())
    if total_docs == 0:
        return {}

    # Document frequency across all tweets (for IDF)
    df: Counter = Counter()
    for texts in clusters.values():
        for text in texts:
            for word in set(_tokenize(text)):
                df[word] += 1

    labels: Dict[int, str] = {}
    for community_id, texts in clusters.items():
        tf: Counter = Counter()
        for text in texts:
            tf.update(_tokenize(text))

        # TF-IDF: penalise words that appear in many clusters
        scored = {
            word: (count / max(len(texts), 1))
            * math.log((total_docs + 1) / (df[word] + 1))
            for word, count in tf.items()
        }
        top_words = sorted(scored, key=scored.get, reverse=True)[:top_n]
        labels[community_id] = " · ".join(top_words) if top_words else f"group {community_id}"

    return labels


def save_categories(partition: Dict[str, int], labels: Dict[int, str]):
    counts = Counter(partition.values())
    data = {
        "partition": partition,
        "labels": {str(k): v for k, v in labels.items()},
        "counts": {str(k): int(v) for k, v in counts.items()},
    }
    CATEGORIES_PATH.write_text(json.dumps(data, indent=2))


def load_categories() -> dict:
    if not CATEGORIES_PATH.exists():
        raise FileNotFoundError("Categories not built yet. Run: bookmarks build")
    return json.loads(CATEGORIES_PATH.read_text())
