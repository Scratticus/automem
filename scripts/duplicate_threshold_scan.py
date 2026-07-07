#!/usr/bin/env python3
"""Duplicate-threshold calibration: pairwise similarity across a live corpus.

Read-only companion to the strict-mode duplicate gate. Scrolls every vector out
of Qdrant, computes the full pairwise cosine-similarity matrix, and reports the
distribution plus the most-similar pairs by memory name. Calibration evidence for
MEMORY_DUPLICATE_SUSPECT_FLOOR, and the periodic audit for differently-named
duplicates the name-collision gate cannot see (legitimate siblings interleave
with genuine duplicates on similarity, so a human reads this report).

Usage:
  QDRANT_URL=http://host:6333 QDRANT_API_KEY=... duplicate_threshold_scan.py \
      [--collection memories] [--top 40] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

BANDS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]


def scroll_all(url: str, api_key: str, collection: str) -> list[dict]:
    points: list[dict] = []
    offset = None
    while True:
        body: dict = {"limit": 500, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{url}/collections/{collection}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"api-key": api_key, "Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]
        points.extend(result["points"])
        offset = result.get("next_page_offset")
        if offset is None:
            return points


def name_of(payload: dict) -> str:
    content = payload.get("content") or ""
    first = content.splitlines()[0] if content else ""
    return first.split("|")[0].strip() or "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "memories"))
    ap.add_argument("--top", type=int, default=40, help="most-similar pairs to list")
    ap.add_argument("--json", help="write full pair detail above the lowest band here")
    args = ap.parse_args()

    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    if not url:
        print("QDRANT_URL is required", file=sys.stderr)
        return 2

    points = scroll_all(url, api_key, args.collection)
    authored = [
        p for p in points if not (p["payload"].get("content") or "").startswith("Meta-pattern:")
    ]
    print(f"points: {len(points)} total | {len(authored)} authored (system nodes skipped)")

    vectors = np.array([p["vector"] for p in authored], dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1.0)

    # Per-node nearest neighbour: what a store-time duplicate probe would see.
    nearest = sim.max(axis=1)
    print("\nnearest-neighbour similarity distribution (per node):")
    upper = 1.01
    for band in BANDS:
        count = int(((nearest >= band) & (nearest < upper)).sum())
        print(f"  [{band:.2f} – {upper if upper <= 1 else 1.0:.2f})  {count:4d} nodes")
        upper = band
    print(f"  [ below {BANDS[-1]:.2f})  {int((nearest < BANDS[-1]).sum()):4d} nodes")

    # Unique pairs, most similar first.
    iu = np.triu_indices_from(sim, k=1)
    order = np.argsort(sim[iu])[::-1]
    pairs = []
    for idx in order:
        score = float(sim[iu][idx])
        if score < BANDS[-1] and len(pairs) >= args.top:
            break
        a, b = int(iu[0][idx]), int(iu[1][idx])
        pairs.append(
            {
                "similarity": round(score, 4),
                "a": {"id": authored[a]["id"], "name": name_of(authored[a]["payload"])},
                "b": {"id": authored[b]["id"], "name": name_of(authored[b]["payload"])},
            }
        )

    print(f"\ntop {args.top} most-similar pairs:")
    for p in pairs[: args.top]:
        print(f"  {p['similarity']:.4f}  {p['a']['name']}  <->  {p['b']['name']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"nodes": len(authored), "pairs": pairs}, fh, indent=1)
        print(f"\nfull pair detail (down to {BANDS[-1]}): {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
