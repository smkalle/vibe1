"""Twitter API v2 client — fetch bookmarks with pagination."""

from typing import Iterator

import httpx

BASE_URL = "https://api.twitter.com/2"
TWEET_FIELDS = "id,text,author_id,created_at,public_metrics,entities"
USER_FIELDS = "id,name,username"
EXPANSIONS = "author_id"


def get_user_id(token: str) -> str:
    resp = httpx.get(
        f"{BASE_URL}/users/me",
        headers={"Authorization": f"Bearer {token}"},
        params={"user.fields": "id,username"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return data["id"]


def fetch_bookmarks(token: str, user_id: str, max_total: int = 800) -> Iterator[dict]:
    """Yield bookmark dicts, handling pagination. max_total caps total returned."""
    url = f"{BASE_URL}/users/{user_id}/bookmarks"
    params = {
        "tweet.fields": TWEET_FIELDS,
        "user.fields": USER_FIELDS,
        "expansions": EXPANSIONS,
        "max_results": 100,  # API max per page
    }
    users_map: dict = {}
    fetched = 0

    while fetched < max_total:
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()

        # Build author lookup from includes
        for user in body.get("includes", {}).get("users", []):
            users_map[user["id"]] = user

        for tweet in body.get("data", []):
            author = users_map.get(tweet.get("author_id", ""), {})
            username = author.get("username", "unknown")
            yield {
                "id": tweet["id"],
                "text": tweet["text"],
                "author_id": tweet.get("author_id", ""),
                "author_name": author.get("name", ""),
                "author_username": username,
                "created_at": tweet.get("created_at", ""),
                "likes": str(tweet.get("public_metrics", {}).get("like_count", 0)),
                "retweets": str(tweet.get("public_metrics", {}).get("retweet_count", 0)),
                "url": f"https://twitter.com/{username}/status/{tweet['id']}",
            }
            fetched += 1
            if fetched >= max_total:
                return

        next_token = body.get("meta", {}).get("next_token")
        if not next_token:
            break
        params["pagination_token"] = next_token
