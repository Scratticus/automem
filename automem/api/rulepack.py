"""Rule packs rendered from the graph (Tier 1).

The graph is the single source of truth for skills / rule packs: a pack is a
materialization of a topic cluster (anchor + atoms), rendered on demand per
harness. Strict mode (Tier 0) guarantees the atoms parse into their type's
shape; this endpoint is the consumer of that guarantee.

GET /rulepack?tags=<tag>[&tags=...]&format=json|markdown|claude-skill|opencode&name=<artifact>

- Tags match by prefix (a request for topic:writing-style pulls the whole
  subtree), using the tag_prefixes the store already computes.
- The response carries the cluster version (sum of the matched ClusterVersion
  counters), which is the staleness fingerprint clients compare before
  regenerating an artifact.
- Atoms that fail to parse are never silently dropped: they are returned in
  "unrenderable" with their findings, so a pack is honest about its holes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, abort, jsonify, request

from automem.config import MEMORY_TYPES, TYPE_ALIASES
from automem.memory_validation import validate_memory

FORMATS = {"json", "markdown", "claude-skill", "opencode"}

_GROUP_HEADINGS = (
    ("Rules", ("Decision", "Style", "Habit")),
    ("Definitions", ("Context",)),
    ("Preferences", ("Preference",)),
    ("Observations", ("Pattern", "Insight")),
)


# ---------------------------------------------------------------------------
# Pure render core (importable without Flask; the materializer tests use this)
# ---------------------------------------------------------------------------


def build_rulepack(
    nodes: List[Dict[str, Any]],
    *,
    name: str,
    tags: List[str],
    version: int,
    rendered_at: str,
) -> Dict[str, Any]:
    """Parse cluster nodes into the harness-neutral pack structure."""
    atoms: List[Dict[str, Any]] = []
    unrenderable: List[Dict[str, Any]] = []
    for node in nodes:
        content = node.get("content") or ""
        model, findings = validate_memory(
            content,
            node.get("type"),
            [],  # stored tags include server-injected entity:* — not the write gate's business here
            known_types=MEMORY_TYPES,
            type_aliases=TYPE_ALIASES,
        )
        if model is None:
            unrenderable.append(
                {
                    "id": node.get("id"),
                    "first_line": content.splitlines()[0] if content else "",
                    "findings": [f.to_dict() for f in findings],
                }
            )
            continue
        atom = model.transcription()
        atom["id"] = node.get("id")
        atom["importance"] = node.get("importance")
        atoms.append(atom)

    anchors = [a for a in atoms if a.get("anchor")]
    body = [a for a in atoms if not a.get("anchor")]
    body.sort(key=lambda a: (a.get("importance") or 0), reverse=True)

    return {
        "name": name,
        "tags": tags,
        "version": version,
        "rendered_at": rendered_at,
        "anchors": anchors,
        "atoms": body,
        "unrenderable": unrenderable,
    }


def _atom_line(atom: Dict[str, Any]) -> str:
    t = atom.get("type")
    if t in ("Decision", "Style", "Habit"):
        parts = []
        if atom.get("when"):
            parts.append(f"WHEN {atom['when']}")
        parts.append(f"DO {atom['do']}")
        if atom.get("exc"):
            parts.append(f"EXC {atom['exc']}")
        if atom.get("note"):
            parts.append(f"NOTE {atom['note']}")
        return " · ".join(parts)
    if t == "Context":
        return f"{atom.get('defines') or atom.get('anchor')}"
    if t == "Preference":
        return f"PREFERS {atom['prefers']}"
    if t == "Pattern":
        return f"WHEN {atom['when']} RECURS {atom['recurs']}"
    if t == "Insight":
        if atom.get("insight"):
            return atom["insight"]
        return f"EVENT {atom['event']} · EXPECTED {atom['expected']} · RESULT {atom['result']}"
    return ""


def format_markdown(pack: Dict[str, Any]) -> str:
    lines = [f"# {pack['name']}", ""]
    for anchor in pack["anchors"]:
        lines.append(anchor.get("anchor", ""))
    if pack["anchors"]:
        lines.append("")
    for heading, types in _GROUP_HEADINGS:
        group = [a for a in pack["atoms"] if a.get("type") in types]
        if not group:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for atom in group:
            lines.append(f"- {_atom_line(atom)}")
        lines.append("")
    if pack["unrenderable"]:
        lines.append(
            f"<!-- {len(pack['unrenderable'])} atom(s) in this cluster failed to parse "
            "and are NOT rendered; fetch format=json for their findings -->"
        )
        lines.append("")
    lines.append(_version_marker(pack))
    return "\n".join(lines).strip() + "\n"


def _version_marker(pack: Dict[str, Any]) -> str:
    tags = ",".join(pack["tags"])
    return f"<!-- automem-rulepack name={pack['name']} version={pack['version']} tags={tags} -->"


def format_claude_skill(pack: Dict[str, Any]) -> str:
    description = (
        pack["anchors"][0].get("anchor")
        if pack["anchors"]
        else f"Rules rendered from the AutoMem cluster {', '.join(pack['tags'])}"
    )
    frontmatter = f"---\nname: {pack['name']}\ndescription: {description}\n---\n\n"
    return frontmatter + format_markdown(pack)


def format_opencode(pack: Dict[str, Any]) -> str:
    upper = pack["name"].upper()
    return (
        f"<!-- BEGIN AUTOMEM RULEPACK {upper} -->\n"
        + format_markdown(pack)
        + f"<!-- END AUTOMEM RULEPACK {upper} -->\n"
    )


FORMATTERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "markdown": format_markdown,
    "claude-skill": format_claude_skill,
    "opencode": format_opencode,
}


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------


def create_rulepack_blueprint(
    get_memory_graph: Callable[[], Any],
    serialize_node: Callable[[Any], Dict[str, Any]],
    utc_now: Callable[[], str],
    logger: Any,
) -> Blueprint:
    bp = Blueprint("rulepack", __name__)

    @bp.route("/rulepack", methods=["GET"])
    def rulepack() -> Any:
        raw = request.args.getlist("tags") or []
        tags = [t.strip().lower() for t in raw if t and t.strip()]
        if not tags:
            abort(400, description="'tags' query parameter is required")
        fmt = request.args.get("format", "json")
        if fmt not in FORMATS:
            abort(400, description=f"'format' must be one of: {', '.join(sorted(FORMATS))}")
        name = request.args.get("name") or tags[0].split(":")[-1]

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        result = graph.query(
            """
            MATCH (m:Memory)
            WHERE ANY(p IN coalesce(m.tag_prefixes, coalesce(m.tags, [])) WHERE p IN $tags)
            RETURN m
            ORDER BY m.importance DESC
            LIMIT 500
            """,
            {"tags": tags},
        )
        nodes = [serialize_node(row[0]) for row in getattr(result, "result_set", []) or []]

        version = 0
        try:
            vres = graph.query(
                "MATCH (v:ClusterVersion) WHERE ANY(p IN $tags WHERE v.tag STARTS WITH p) "
                "RETURN sum(v.version)",
                {"tags": tags},
            )
            rows = getattr(vres, "result_set", None)
            if rows and rows[0] and rows[0][0]:
                version = int(rows[0][0])
        except Exception:
            logger.exception("ClusterVersion lookup failed; version stays 0")

        pack = build_rulepack(nodes, name=name, tags=tags, version=version, rendered_at=utc_now())
        if fmt == "json":
            return jsonify(pack)
        return FORMATTERS[fmt](pack), 200, {"Content-Type": "text/markdown; charset=utf-8"}

    return bp
