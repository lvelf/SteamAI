import argparse
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

STEAM_REVIEWS_ENDPOINT = "https://store.steampowered.com/appreviews/{appid}"


@dataclass
class FetchResult:
    reviews: List[Dict[str, Any]]
    newest_ts_updated: int
    pages_fetched: int
    stopped_reason: str


def _request_with_backoff(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    *,
    timeout_s: int = 30,
    max_retries: int = 8,
    base_sleep_s: float = 1.0,
    max_sleep_s: float = 60.0,
) -> Dict[str, Any]:
    """
    GET:
    - 429: retry
    - 5xx: retry
    - 4xx: direct raise
    """
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout_s)

            if r.status_code == 429 or 500 <= r.status_code < 600:
                # jitter
                sleep = min(max_sleep_s, base_sleep_s * (2 ** attempt))
                sleep *= (0.7 + random.random() * 0.6)  # jitter: [0.7, 1.3)
                time.sleep(sleep)
                continue

            # other 4xx
            r.raise_for_status()
            return r.json()

        except (requests.Timeout, requests.ConnectionError, requests.ChunkedEncodingError) as e:
            last_exc = e
            sleep = min(max_sleep_s, base_sleep_s * (2 ** attempt))
            sleep *= (0.7 + random.random() * 0.6)
            time.sleep(sleep)
            continue
        except requests.HTTPError as e:
            # !429 的 4xx other HTTPError
            raise

    # retry times go 
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed after retries")


def fetch_reviews_incremental(
    appid: int,
    since_ts_updated: int,
    *,
    max_pages: int = 20,
    sleep_s: float = 1.0,
    timeout_s: int = 30,
    max_retries: int = 8,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 60.0,
    filter_mode: str = "updated",   # "updated" / "recent" / "all"
    language: str = "all",
    verbose: bool = False,
) -> FetchResult:
    """
    incremental fetch:
    only fetch timestamp_updated <= since_ts_updated 
    cursor loop 
    """
    session = requests.Session()

    cursor = "*"
    out: List[Dict[str, Any]] = []
    newest_seen = int(since_ts_updated)

    seen_cursors = set()
    seen_reco_ids = set()

    pages = 0
    stopped = "max_pages_reached"

    for page_idx in range(max_pages):
        pages = page_idx + 1

        # cursor loop check
        if cursor in seen_cursors:
            stopped = f"cursor_loop_detected(cursor={cursor})"
            break
        seen_cursors.add(cursor)

        params = {
            "json": 1,
            "filter": filter_mode,
            "language": language,
            "num_per_page": 100,
            "cursor": cursor,
        }

        url = STEAM_REVIEWS_ENDPOINT.format(appid=appid)
        data = _request_with_backoff(
            session,
            url,
            params,
            timeout_s=timeout_s,
            max_retries=max_retries,
            base_sleep_s=base_backoff_s,
            max_sleep_s=max_backoff_s,
        )

        reviews = data.get("reviews", []) or []
        next_cursor = data.get("cursor")

        if verbose:
            q_summary = data.get("query_summary", {})
            total = q_summary.get("total_reviews")
            print(f"[page {pages}] got={len(reviews)} next_cursor={'yes' if next_cursor else 'no'} total={total}")

        if not reviews:
            stopped = "empty_reviews"
            break

        # incremental + reduce dup
        new_in_page = 0
        saw_old = False

        for rv in reviews:
            rid = str(rv.get("recommendationid", ""))
            if not rid or rid in seen_reco_ids:
                continue
            seen_reco_ids.add(rid)

            tsu = int(rv.get("timestamp_updated", 0))
            if tsu <= since_ts_updated:
                saw_old = True
                continue

            out.append(rv)
            new_in_page += 1
            if tsu > newest_seen:
                newest_seen = tsu

        # if not new, then old
        if saw_old:
            stopped = "hit_old_data"
            break

        # next page
        if not next_cursor:
            stopped = "no_next_cursor"
            break

        cursor = next_cursor

        # genral sleep
        if sleep_s > 0:
            time.sleep(sleep_s)

    return FetchResult(
        reviews=out,
        newest_ts_updated=newest_seen,
        pages_fetched=pages,
        stopped_reason=stopped,
    )


def _main():
    parser = argparse.ArgumentParser(description="Steam reviews fetch test")
    parser.add_argument("--appid", type=int, required=True, help="Steam appid, e.g. 620 (Portal 2)")
    parser.add_argument("--since", type=int, default=0, help="since timestamp_updated (unix seconds)")
    parser.add_argument("--pages", type=int, default=5, help="max pages to fetch")
    parser.add_argument("--sleep", type=float, default=1.0, help="sleep seconds between pages")
    parser.add_argument("--filter", type=str, default="updated", choices=["updated", "recent", "all"])
    parser.add_argument("--lang", type=str, default="all", help="language: all / english / etc")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--base-backoff", type=float, default=1.0)
    parser.add_argument("--max-backoff", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    res = fetch_reviews_incremental(
        appid=args.appid,
        since_ts_updated=args.since,
        max_pages=args.pages,
        sleep_s=args.sleep,
        timeout_s=args.timeout,
        max_retries=args.retries,
        base_backoff_s=args.base_backoff,
        max_backoff_s=args.max_backoff,
        filter_mode=args.filter,
        language=args.lang,
        verbose=args.verbose,
    )

    print("\n==== Fetch Summary ====")
    print(f"appid: {args.appid}")
    print(f"since_ts_updated: {args.since}")
    print(f"pages_fetched: {res.pages_fetched}")
    print(f"fetched_reviews: {len(res.reviews)}")
    print(f"newest_ts_updated: {res.newest_ts_updated}")
    print(f"stopped_reason: {res.stopped_reason}")

    for i, rv in enumerate(res.reviews[:3]):
        print(f"\n--- sample #{i+1} ---")
        print("recommendationid:", rv.get("recommendationid"))
        print("timestamp_updated:", rv.get("timestamp_updated"))
        print("voted_up:", rv.get("voted_up"))
        txt = (rv.get("review") or "").replace("\n", " ")
        print("review_snippet:", txt[:200])


if __name__ == "__main__":
    _main()
