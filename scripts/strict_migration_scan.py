#!/usr/bin/env python3
"""Strict-mode migration scanner: audit an existing graph for gate compatibility.

Runs the same validation the strict-mode write gate applies (automem.memory_validation)
over every node in a live graph, read-only, and reports what would fail and how to fix
it. This is the upgrade path for a corpus that predates strict mode: run the scan,
remediate, enable the flag.

System-generated nodes (enrichment meta-patterns) are counted separately — they carry
authored-memory types but were never authored, so they are not remediation work.

Usage:
  strict_migration_scan.py [--endpoint URL] [--json OUT.json] [--top N]

Token: AUTOMEM_TOKEN env var, else keyring automem/api_token.
Contributor names (attribution check) come from MEMORY_STRICT_CONTRIBUTOR_NAMES,
the same env the write gate reads; unset = attribution is not audited.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automem.config import MEMORY_STRICT_CONTRIBUTOR_NAMES, MEMORY_TYPES, TYPE_ALIASES  # noqa: E402
from automem.memory_validation import validate_memory  # noqa: E402

REMEDIATION = {
    "secret-material": "URGENT: remove the credential; reference its location instead",
    "memory-format/top-line": "add the '{name} | (scope:{x} |) tier:{n}' top line",
    "type-enum": "retype to a canonical type",
    "name-encodes-type": "rename; the type field carries the type",
    "memory-format/why-clause": "fold the rationale into WHEN/DO or drop it",
    "memory-format/unparsed-line": "rework the line into the type's shape, or split it out",
    "enumeration-markers": "eject the example into its own memory via EXEMPLIFIES",
    "atomicity": "split the bundle into single-concept memories",
    "attribution-in-content": "remove the contributor name; provenance lives in entity tags",
}
SHAPE_FIX = "rework the body to the type's shape"


def token() -> str:
    tok = os.environ.get("AUTOMEM_TOKEN")
    if tok:
        return tok
    try:
        import keyring

        return keyring.get_password("automem", "api_token") or ""
    except ImportError:
        # venv without keyring: the system Python owns the system-managed keyring
        import subprocess

        result = subprocess.run(
            [
                "python3",
                "-c",
                "import keyring; print(keyring.get_password('automem', 'api_token') or '')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint", default=os.environ.get("AUTOMEM_API_URL", "http://127.0.0.1:8001")
    )
    ap.add_argument("--json", help="write the full per-node detail to this path")
    ap.add_argument("--top", type=int, default=15, help="offenders to list, by importance")
    args = ap.parse_args()

    req = urllib.request.Request(
        f"{args.endpoint}/graph/snapshot?limit=5000&min_importance=0",
        headers={"Authorization": f"Bearer {token()}"},
    )
    nodes = json.loads(urllib.request.urlopen(req, timeout=60).read())["nodes"]

    system_nodes = 0
    clean = 0
    offenders = []
    check_counts: collections.Counter = collections.Counter()

    for node in nodes:
        content = node.get("content") or ""
        if content.startswith("Meta-pattern:"):
            system_nodes += 1
            continue
        tags = [
            t for t in (node.get("tags") or []) if not t.lower().startswith(("entity:", "person:"))
        ]
        _, findings = validate_memory(
            content,
            node.get("type"),
            [],  # stored entity tags are server-injected; tag hygiene is a write-time concern
            known_types=MEMORY_TYPES,
            type_aliases=TYPE_ALIASES,
            contributor_names=MEMORY_STRICT_CONTRIBUTOR_NAMES,
        )
        rejections = [f for f in findings if f.severity == "reject"]
        if not rejections:
            clean += 1
            continue
        for f in rejections:
            check_counts[f.check] += 1
        name = content.splitlines()[0].split("|")[0].strip() if content else node.get("id", "?")
        offenders.append(
            {
                "id": node.get("id"),
                "name": name,
                "importance": node.get("importance") or 0,
                "type": node.get("type"),
                "tags": tags,
                "checks": sorted({f.check for f in rejections}),
                "remediation": sorted(
                    {
                        REMEDIATION.get(
                            f.check, SHAPE_FIX if f.check.startswith("memory-format/") else f.check
                        )
                        for f in rejections
                    }
                ),
                "findings": [f.to_dict() for f in rejections],
            }
        )

    offenders.sort(key=lambda o: o["importance"], reverse=True)
    authored = clean + len(offenders)

    print(
        f"nodes: {len(nodes)} total | {system_nodes} system-generated (skipped) | {authored} authored"
    )
    print(
        "contributor names enforced: "
        + (", ".join(MEMORY_STRICT_CONTRIBUTOR_NAMES) or "(none — attribution check INACTIVE)")
    )
    print(
        f"authored memories passing the strict gate: {clean} ({clean * 100 // max(authored, 1)}%)"
    )
    print(f"authored memories needing remediation:     {len(offenders)}")
    print("\nfindings by check:")
    for check, count in check_counts.most_common():
        print(f"  {count:4d}  {check}  ->  {REMEDIATION.get(check, SHAPE_FIX)}")
    print(f"\ntop offenders by importance (first {args.top}):")
    for o in offenders[: args.top]:
        print(f"  [{o['importance']:.2f}] {o['name']}: {', '.join(o['checks'])}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"summary": dict(check_counts), "offenders": offenders}, fh, indent=1)
        print(f"\nfull detail: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
