from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from flask import Blueprint, abort, jsonify, make_response, request
from flask.typing import ResponseReturnValue

from automem.config import (
    CLASSIFICATION_MODEL,
    MEMORY_AUTHORING_STANDARD_FILE,
    MEMORY_AUTO_SUMMARIZE,
    MEMORY_CONTENT_HARD_LIMIT,
    MEMORY_CONTENT_SOFT_LIMIT,
    MEMORY_DUPLICATE_LOG_TAG,
    MEMORY_DUPLICATE_SUSPECT_FLOOR,
    MEMORY_DUPLICATE_SUSPECT_LIMIT,
    MEMORY_STRICT_CONTRIBUTOR_NAMES,
    MEMORY_STRICT_VALIDATION,
    MEMORY_SUMMARY_TARGET_LENGTH,
    MEMORY_TYPES,
    TYPE_ALIASES,
    normalize_memory_type,
)
from automem.memory_validation import (
    Finding,
    authoring_standard,
    memory_name,
    rejection_body,
    split_findings,
    validate_memory,
)
from automem.utils.text import should_summarize_content, summarize_content


def _strict_gate(
    content: str, memory_type: Optional[str], tags: List[str]
) -> tuple[Optional[str], List[Dict[str, str]]]:
    """Run strict-mode validation for one memory write.

    Returns (canonical_type, warnings) — the type comes back alias-normalized so
    strict instances only ever store canonical types. Aborts the request with a
    400 findings body (carrying the authoring standard) on any rejection.
    """
    # Normalize aliases BEFORE validating so the shape check runs against the
    # canonical type ("decision" must be held to the Decision schema, not skipped).
    canonical = memory_type
    if memory_type:
        normalized, _ = normalize_memory_type(memory_type)
        if normalized:
            canonical = normalized
    _, findings = validate_memory(
        content,
        canonical,
        tags,
        known_types=MEMORY_TYPES,
        type_aliases=TYPE_ALIASES,
        contributor_names=MEMORY_STRICT_CONTRIBUTOR_NAMES,
    )
    rejections, warnings = split_findings(findings)
    if rejections:
        body = rejection_body(
            rejections, warnings, authoring_standard(MEMORY_AUTHORING_STANDARD_FILE)
        )
        abort(make_response(jsonify(body), 400))
    return canonical, [w.to_dict() for w in warnings]


def _is_log_class(tags_lower: List[str]) -> bool:
    """Record-class memories (event logs) legitimately share names and phrasing."""
    return MEMORY_DUPLICATE_LOG_TAG in tags_lower


def find_name_collision(
    graph: Any, content: str, exclude_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the existing memory whose top-line name matches *content*'s, if any.

    Names are identifiers ([[name]] links, skill routing, the renderer) — a second
    memory under an existing name is a duplicate by definition. Corpus calibration
    (2026-07-07) showed this is the reliable duplicate signal; cosine similarity
    interleaves genuine duplicates with legitimate siblings and must stay advisory.
    """
    name = memory_name(content)
    if not name:
        return None
    result = graph.query(
        "MATCH (m:Memory) WHERE m.content STARTS WITH $p1 OR m.content STARTS WITH $p2 "
        "RETURN m.id, m.content, m.type, m.tags LIMIT 10",
        {"p1": f"{name} |", "p2": f"{name}|"},
    )
    for row in getattr(result, "result_set", None) or []:
        row_id, row_content, row_type = str(row[0]), row[1] or "", row[2]
        row_tags = [str(t).lower() for t in (row[3] or [])]
        if exclude_id and row_id == str(exclude_id):
            continue
        if memory_name(row_content) != name:
            continue
        if _is_log_class(row_tags):
            # Existing log-class entries share names by design (one entry per
            # event); they never block, and new entries exempt themselves by
            # carrying the tag — checked by the caller before this lookup.
            continue
        return {"id": row_id, "content": row_content, "type": row_type}
    return None


def duplicate_name_finding(name: Optional[str], existing: Dict[str, Any]) -> Finding:
    return Finding(
        "duplicate-name",
        "reject",
        f"A memory named {name!r} already exists (id {existing['id']}, "
        f"type {existing['type']}). Update the existing memory (PATCH) or choose a "
        f"genuinely different name. Deliberate record-class memories that share names "
        f"(event logs) must carry the {MEMORY_DUPLICATE_LOG_TAG!r} tag.",
    )


def find_duplicate_suspects(
    qdrant_client: Any,
    collection_name: str,
    vector: List[float],
    exclude_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """ADVISORY nearest-neighbour probe: the memories most similar to a draft.

    Returned on store/validate responses for a human (via the harness linter) to
    judge — never a rejection: legitimate siblings can outscore genuine duplicates.
    Best-effort; any failure returns [] and never blocks the write.
    """
    try:
        hits = qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=MEMORY_DUPLICATE_SUSPECT_LIMIT + (1 if exclude_id else 0),
            with_payload=True,
        )
    except Exception:
        return []
    suspects = []
    for hit in hits:
        hit_id = str(hit.id)
        if exclude_id and hit_id == str(exclude_id):
            continue
        score = float(hit.score)
        if score < MEMORY_DUPLICATE_SUSPECT_FLOOR:
            continue
        payload = getattr(hit, "payload", None) or {}
        suspects.append(
            {
                "id": hit_id,
                "name": memory_name(payload.get("content") or "") or "?",
                "similarity": round(score, 4),
            }
        )
    return suspects[:MEMORY_DUPLICATE_SUSPECT_LIMIT]


def _validate_memory_id(memory_id: str) -> None:
    """Abort with 400 if *memory_id* is not a valid UUID."""
    try:
        uuid.UUID(memory_id)
    except ValueError:
        abort(400, description="memory_id must be a valid UUID")


def _uuid_error(memory_id: str, field_name: str) -> Optional[str]:
    try:
        uuid.UUID(memory_id)
    except ValueError:
        return f"'{field_name}' must be a valid UUID"
    return None


def _association_summary(created_count: int, total_count: int) -> str:
    return f"{created_count}/{total_count} associations created successfully"


def _association_failure(index: int, reason: str) -> Dict[str, Any]:
    return {"index": index, "reason": reason}


def _association_success(
    *,
    index: int,
    memory1_id: str,
    memory2_id: str,
    relation_type: str,
    strength: float,
) -> Dict[str, Any]:
    return {
        "index": index,
        "memory1_id": memory1_id,
        "memory2_id": memory2_id,
        "relation_type": relation_type,
        "strength": strength,
    }


def _prepare_association_props(
    *,
    payload: Dict[str, Any],
    relation_type: str,
    strength: float,
    timestamp: str,
    relationship_types: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    relationship_props = {"strength": strength, "updated_at": timestamp}
    relation_config = relationship_types.get(relation_type, {})
    for prop in relation_config.get("properties", []):
        if prop in payload and prop not in relationship_props:
            relationship_props[prop] = payload[prop]
    return relationship_props


def _parse_association_item(
    *,
    item: Any,
    index: int,
    authorable_relations: Set[str],
    relationship_types: Dict[str, Dict[str, Any]],
    coerce_importance_fn: Callable[[Any], float],
    timestamp: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not isinstance(item, dict):
        return None, _association_failure(index, "Association item must be an object")

    memory1_id = str(item.get("memory1_id") or "").strip()
    memory2_id = str(item.get("memory2_id") or "").strip()
    relation_type = str(item.get("type") or "RELATES_TO").strip().upper()
    strength = coerce_importance_fn(item.get("strength", 0.5))

    if not memory1_id or not memory2_id:
        return None, _association_failure(index, "'memory1_id' and 'memory2_id' are required")

    for field_name, value in (("memory1_id", memory1_id), ("memory2_id", memory2_id)):
        error = _uuid_error(value, field_name)
        if error:
            return None, _association_failure(index, error)

    if memory1_id == memory2_id:
        return None, _association_failure(index, "Cannot associate a memory with itself")

    if relation_type not in authorable_relations:
        return None, _association_failure(
            index,
            f"Relation type must be one of {sorted(authorable_relations)}",
        )

    return (
        {
            "index": index,
            "memory1_id": memory1_id,
            "memory2_id": memory2_id,
            "type": relation_type,
            "strength": strength,
            "props": _prepare_association_props(
                payload=item,
                relation_type=relation_type,
                strength=strength,
                timestamp=timestamp,
                relationship_types=relationship_types,
            ),
        },
        None,
    )


def _batch_association_response(
    *,
    succeeded: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
    total_count: int,
    jsonify_fn: Callable[[Any], Any],
) -> Any:
    created_count = len(succeeded)
    failed_count = len(failed)
    status_code = 201 if failed_count == 0 else 207
    status = "success" if failed_count == 0 else "partial_success"
    return (
        jsonify_fn(
            {
                "status": status,
                "created_count": created_count,
                "failed_count": failed_count,
                "succeeded": sorted(succeeded, key=lambda item: item["index"]),
                "failed": sorted(failed, key=lambda item: item["index"]),
                "summary": _association_summary(created_count, total_count),
            }
        ),
        status_code,
    )


def _create_association_batch(
    *,
    payload: Dict[str, Any],
    coerce_importance_fn: Callable[[Any], float],
    get_memory_graph_fn: Callable[[], Any],
    authorable_relations: Set[str],
    relationship_types: Dict[str, Dict[str, Any]],
    utc_now_fn: Callable[[], str],
    abort_fn: Callable[..., Any],
    jsonify_fn: Callable[[Any], Any],
    logger: Any,
) -> Any:
    associations = payload.get("associations")
    if not isinstance(associations, list) or len(associations) == 0:
        abort_fn(400, description="'associations' must be a non-empty array")
    if len(associations) > 500:
        abort_fn(400, description="Batch size limit is 500 associations per request")

    timestamp = utc_now_fn()
    valid_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for index, item in enumerate(associations):
        row, failure = _parse_association_item(
            item=item,
            index=index,
            authorable_relations=authorable_relations,
            relationship_types=relationship_types,
            coerce_importance_fn=coerce_importance_fn,
            timestamp=timestamp,
        )
        if failure:
            failed.append(failure)
        elif row:
            valid_rows.append(row)

    succeeded: List[Dict[str, Any]] = []
    if valid_rows:
        graph = get_memory_graph_fn()
        if graph is None:
            abort_fn(503, description="FalkorDB is unavailable")

        rows_by_index = {row["index"]: row for row in valid_rows}
        rows_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for row in valid_rows:
            rows_by_type.setdefault(row["type"], []).append(row)

        for relation_type, rows in rows_by_type.items():
            try:
                result = graph.query(
                    f"""
                    UNWIND $rows AS row
                    MATCH (m1:Memory {{id: row.memory1_id}})
                    MATCH (m2:Memory {{id: row.memory2_id}})
                    MERGE (m1)-[r:{relation_type}]->(m2)
                    SET r += row.props
                    RETURN row.index, row.memory1_id, row.memory2_id
                    """,
                    {"rows": rows},
                )
            except Exception:
                logger.exception(
                    "Failed to create association batch for relation type %s",
                    relation_type,
                )
                for row in rows:
                    failed.append(
                        _association_failure(
                            row["index"],
                            f"Failed to create association batch for relation type {relation_type}",
                        )
                    )
                continue

            created_indexes = set()
            for result_row in list(getattr(result, "result_set", []) or []):
                index = int(result_row[0])
                created_indexes.add(index)
                source = rows_by_index[index]
                succeeded.append(
                    _association_success(
                        index=index,
                        memory1_id=source["memory1_id"],
                        memory2_id=source["memory2_id"],
                        relation_type=source["type"],
                        strength=source["strength"],
                    )
                )

            for row in rows:
                if row["index"] not in created_indexes:
                    failed.append(
                        _association_failure(row["index"], "One or both memories do not exist")
                    )

    return _batch_association_response(
        succeeded=succeeded,
        failed=failed,
        total_count=len(associations),
        jsonify_fn=jsonify_fn,
    )


def _parse_by_tag_request(
    *,
    request_args: Any,
    normalize_tag_list_fn: Callable[[Any], List[str]],
    abort_fn: Callable[..., Any],
) -> tuple[List[str], int, int]:
    raw_tags = request_args.getlist("tags") or request_args.get("tags")
    tags = normalize_tag_list_fn(raw_tags)
    if not tags:
        abort_fn(400, description="'tags' query parameter is required")

    try:
        limit = int(request_args.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 200))

    try:
        offset = int(request_args.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)

    return tags, limit, offset


def _load_memories_by_tag_page(
    *,
    graph: Any,
    tags: List[str],
    limit: int,
    offset: int,
    serialize_node_fn: Callable[[Any], Dict[str, Any]],
    parse_metadata_field_fn: Callable[[Any], Any],
    logger: Any,
    abort_fn: Callable[..., Any],
) -> tuple[List[Dict[str, Any]], bool]:
    params = {
        "tags": [tag.lower() for tag in tags],
        "offset": offset,
        "limit_plus_one": limit + 1,
    }
    query = """
        MATCH (m:Memory)
        WHERE ANY(tag IN coalesce(m.tags, []) WHERE toLower(tag) IN $tags)
        RETURN m
        ORDER BY m.importance DESC, m.timestamp DESC, m.id ASC
        SKIP $offset
        LIMIT $limit_plus_one
    """
    try:
        result = graph.query(query, params)
    except Exception:
        logger.exception("Tag search failed")
        abort_fn(500, description="Failed to search by tag")

    rows = list(getattr(result, "result_set", []) or [])
    has_more = len(rows) > limit
    memories: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        data = serialize_node_fn(row[0])
        data["metadata"] = parse_metadata_field_fn(data.get("metadata"))
        memories.append(data)

    return memories, has_more


def _delete_qdrant_points(
    *,
    qdrant_client: Any,
    collection_name: str,
    memory_ids: List[str],
    logger: Any,
) -> None:
    if qdrant_client is None or not memory_ids:
        return

    selector: Any = {"points": memory_ids}
    try:
        from qdrant_client.http import models as http_models  # type: ignore

        selector = http_models.PointIdsList(points=memory_ids)
    except Exception:
        pass

    try:
        qdrant_client.delete(collection_name=collection_name, points_selector=selector)
    except Exception:
        logger.exception("Failed to delete vectors for %d memories", len(memory_ids))


def _delete_graph_memories(
    *,
    graph: Any,
    memory_ids: List[str],
    logger: Any,
    abort_fn: Any,
) -> None:
    if not memory_ids:
        return

    try:
        graph.query("MATCH (m:Memory) WHERE m.id IN $ids DETACH DELETE m", {"ids": memory_ids})
    except Exception:
        logger.exception("Bulk delete by tag failed for %d memories", len(memory_ids))
        abort_fn(500, description="Failed to delete memories by tag")


def create_memory_blueprint(
    store_memory: Callable[[], Any],
    update_memory: Callable[[str], Any],
    delete_memory: Callable[[str], Any],
    by_tag: Callable[[], Any],
    associate: Callable[[], Any],
    delete_by_tag: Optional[Callable[[], Any]] = None,
) -> Blueprint:
    """Compatibility wrapper around the legacy handlers in app.py."""
    bp = Blueprint("memory", __name__)

    @bp.route("/memory", methods=["POST"])
    def _store() -> Any:
        return store_memory()

    @bp.route("/memory/<memory_id>", methods=["PATCH"])
    def _update(memory_id: str) -> Any:
        return update_memory(memory_id)

    @bp.route("/memory/<memory_id>", methods=["DELETE"])
    def _delete(memory_id: str) -> Any:
        return delete_memory(memory_id)

    @bp.route("/memory/by-tag", methods=["GET", "DELETE"])
    def _by_tag() -> Any:
        if request.method == "DELETE":
            if delete_by_tag is None:
                return by_tag()
            return delete_by_tag()
        return by_tag()

    @bp.route("/associate", methods=["POST"])
    def _associate() -> Any:
        return associate()

    return bp


def create_memory_blueprint_full(
    get_memory_graph: Callable[[], Any],
    get_qdrant_client: Callable[[], Any],
    normalize_tags: Callable[[Any], List[str]],
    normalize_tag_list: Callable[[Any], List[str]],
    compute_tag_prefixes: Callable[[List[str]], List[str]],
    coerce_importance: Callable[[Any], float],
    coerce_embedding: Callable[[Any], Optional[List[float]]],
    normalize_timestamp: Callable[[str], str],
    utc_now: Callable[[], str],
    serialize_node: Callable[[Any], Dict[str, Any]],
    parse_metadata_field: Callable[[Any], Any],
    generate_real_embedding: Callable[[str], List[float]],
    enqueue_enrichment: Callable[[str], None],
    enqueue_embedding: Callable[[str, str], None],
    memory_classify: Callable[[str], tuple[str, float]],
    point_struct: Any,
    collection_name: str,
    authorable_relations: Set[str] | List[str],
    relation_types: Dict[str, Any],
    state: Any,
    logger: Any,
    on_access: Optional[Callable[[List[str]], None]] = None,
    get_openai_client: Optional[Callable[[], Any]] = None,
    generate_real_embeddings_batch: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> Blueprint:
    bp = Blueprint("memory", __name__)

    def _bump_cluster_versions(graph: Any, tags: List[str]) -> None:
        """Strict mode: bump the per-tag cluster version on any accepted write.

        The counters are the staleness signal for lazily-regenerated artifacts
        (rendered rule packs / skills). Best-effort — never blocks the write.
        """
        if not tags:
            return
        try:
            graph.query(
                "UNWIND $tags AS t MERGE (v:ClusterVersion {tag: t}) "
                "SET v.version = coalesce(v.version, 0) + 1, v.updated_at = $now",
                {"tags": tags, "now": utc_now()},
            )
        except Exception:
            logger.exception("Cluster version bump failed")

    @bp.route("/memory/validate", methods=["POST"])
    def validate() -> Any:
        """Dry-run the write gate: return the findings for a draft memory, write nothing.

        This is what client preflights call instead of carrying their own copy
        of the rules — one rule source, no client drift. Always 200; an empty
        findings list means the draft would be accepted.
        """
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")
        raw_type = payload.get("type")
        canonical = raw_type
        if raw_type:
            normalized, _ = normalize_memory_type(raw_type)
            if normalized:
                canonical = normalized
        content = (payload.get("content") or "").strip()
        tags = normalize_tags(payload.get("tags"))
        _, findings = validate_memory(
            content,
            canonical,
            tags,
            known_types=MEMORY_TYPES,
            type_aliases=TYPE_ALIASES,
            contributor_names=MEMORY_STRICT_CONTRIBUTOR_NAMES,
        )
        response: Dict[str, Any] = {
            "findings": [f.to_dict() for f in findings],
            "canonical_type": canonical,
            "strict_enforced": MEMORY_STRICT_VALIDATION,
        }
        # Duplicate checks mirror the store gate so preflights teach BEFORE the
        # store attempt. The exclude_id param lets an update preflight skip the
        # memory it is editing.
        tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        exclude_id = payload.get("exclude_id")
        if content and not _is_log_class(tags_lower):
            graph = get_memory_graph()
            if graph is not None:
                existing = find_name_collision(graph, content, exclude_id=exclude_id)
                if existing:
                    response["findings"].append(
                        duplicate_name_finding(memory_name(content), existing).to_dict()
                    )
                    response["existing_memory"] = existing
            probe_client = get_qdrant_client()
            if probe_client is not None:
                try:
                    vector = generate_real_embedding(content)
                except Exception:
                    vector = None
                if vector is not None:
                    response["duplicate_suspects"] = find_duplicate_suspects(
                        probe_client, collection_name, vector, exclude_id=exclude_id
                    )
        if response["findings"]:
            # Mirror the 400 gate: a failed preflight carries the standard, so a
            # thin client attaches it to its deny without holding a local copy —
            # the server file is the ONLY copy anywhere.
            response["authoring_standard"] = authoring_standard(MEMORY_AUTHORING_STANDARD_FILE)
        return jsonify(response)

    @bp.route("/memory", methods=["POST"])
    def store() -> Any:
        query_start = time.perf_counter()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")

        content = (payload.get("content") or "").strip()
        if not content:
            abort(400, description="'content' is required")

        # Content size governance: check if summarization or rejection is needed
        original_content: Optional[str] = None
        content_action = should_summarize_content(
            content, MEMORY_CONTENT_SOFT_LIMIT, MEMORY_CONTENT_HARD_LIMIT
        )

        if content_action == "reject":
            abort(
                400,
                description=f"Content exceeds maximum length of {MEMORY_CONTENT_HARD_LIMIT} characters "
                f"({len(content)} provided). Please split into smaller memories or summarize.",
            )

        if content_action == "summarize" and MEMORY_AUTO_SUMMARIZE and not MEMORY_STRICT_VALIDATION:
            # Strict mode never auto-summarizes: a silent LLM rewrite would break
            # the very shape guarantees the mode enforces. Oversized-but-valid
            # content comes back as a warning instead.
            openai_client = get_openai_client() if get_openai_client else None
            if openai_client:
                summary = summarize_content(
                    content,
                    openai_client,
                    CLASSIFICATION_MODEL,
                    MEMORY_SUMMARY_TARGET_LENGTH,
                )
                if summary:
                    original_content = content
                    content = summary
                    logger.info(
                        "Auto-summarized oversized memory: %d -> %d chars",
                        len(original_content),
                        len(content),
                    )
                else:
                    logger.warning(
                        "Auto-summarization failed for %d char content, storing as-is",
                        len(content),
                    )
            else:
                logger.warning(
                    "Content exceeds soft limit (%d chars) but OpenAI client unavailable for summarization",
                    len(content),
                )

        tags = normalize_tags(payload.get("tags"))
        tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        tag_prefixes = compute_tag_prefixes(tags_lower)
        importance = coerce_importance(payload.get("importance"))
        # Always generate server-side UUID to prevent collision/overwrite attacks
        memory_id = str(uuid.uuid4())

        metadata_raw = payload.get("metadata")
        if metadata_raw is None:
            metadata: Dict[str, Any] = {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            abort(400, description="'metadata' must be an object")

        # If content was summarized, preserve original in metadata for audit trail
        if original_content:
            metadata["original_content"] = original_content
            metadata["was_summarized"] = True
            metadata["original_length"] = len(original_content)

        metadata_json = json.dumps(metadata, default=str)

        # Accept explicit type/confidence or classify automatically
        memory_type = payload.get("type")
        type_confidence = payload.get("confidence")
        if memory_type:
            # Validate explicit type
            # (Memory types are validated by the classifier caller; keep permissive here)
            if type_confidence is None:
                type_confidence = 0.9
            else:
                type_confidence = coerce_importance(type_confidence)
        else:
            memory_type, type_confidence = memory_classify(content)

        strict_warnings: List[Dict[str, str]] = []
        if MEMORY_STRICT_VALIDATION:
            memory_type, strict_warnings = _strict_gate(content, memory_type, tags)
            if len(content) > MEMORY_CONTENT_SOFT_LIMIT:
                strict_warnings.append(
                    {
                        "check": "content-soft-limit",
                        "severity": "warn",
                        "message": f"Content is {len(content)} chars (soft limit "
                        f"{MEMORY_CONTENT_SOFT_LIMIT}); strict mode never auto-summarizes "
                        "— split into atomic memories or tighten.",
                    }
                )

        t_valid = payload.get("t_valid")
        t_invalid = payload.get("t_invalid")
        if t_valid:
            try:
                t_valid = normalize_timestamp(t_valid)
            except ValueError as exc:
                abort(400, description=f"Invalid t_valid: {exc}")
        if t_invalid:
            try:
                t_invalid = normalize_timestamp(t_invalid)
            except ValueError as exc:
                abort(400, description=f"Invalid t_invalid: {exc}")

        try:
            embedding = coerce_embedding(payload.get("embedding"))
        except ValueError as exc:
            abort(400, description=str(exc))

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        duplicate_suspects: List[Dict[str, Any]] = []
        embedding_generated = False
        if MEMORY_STRICT_VALIDATION and not _is_log_class(tags_lower):
            existing = find_name_collision(graph, content)
            if existing:
                body = rejection_body(
                    [duplicate_name_finding(memory_name(content), existing)],
                    [],
                    authoring_standard(MEMORY_AUTHORING_STANDARD_FILE),
                )
                body["existing_memory"] = existing
                abort(make_response(jsonify(body), 400))
            probe_client = get_qdrant_client()
            if probe_client is not None:
                if embedding is None:
                    try:
                        embedding = generate_real_embedding(content)
                        embedding_generated = True
                    except Exception:
                        logger.exception("Embedding for duplicate probe failed; advisory skipped")
                if embedding is not None:
                    duplicate_suspects = find_duplicate_suspects(
                        probe_client, collection_name, embedding
                    )
                    if duplicate_suspects:
                        strict_warnings.append(
                            {
                                "check": "duplicate-suspects",
                                "severity": "warn",
                                "message": "Nearest existing memories: "
                                + "; ".join(
                                    f"{s['name']} ({s['id']}, {s['similarity']})"
                                    for s in duplicate_suspects
                                )
                                + " — if one of these is the same fact, update it instead "
                                "of storing a variant.",
                            }
                        )

        created_at = payload.get("timestamp")
        if created_at:
            try:
                created_at = normalize_timestamp(created_at)
            except ValueError as exc:
                abort(400, description=str(exc))
        else:
            created_at = utc_now()

        updated_at = payload.get("updated_at")
        if updated_at:
            try:
                updated_at = normalize_timestamp(updated_at)
            except ValueError as exc:
                abort(400, description=f"Invalid updated_at: {exc}")
        else:
            updated_at = created_at

        last_accessed = payload.get("last_accessed")
        if last_accessed:
            try:
                last_accessed = normalize_timestamp(last_accessed)
            except ValueError as exc:
                abort(400, description=f"Invalid last_accessed: {exc}")
        else:
            last_accessed = updated_at

        try:
            graph.query(
                """
                MERGE (m:Memory {id: $id})
                ON CREATE SET
                    m.content = $content,
                    m.timestamp = $timestamp,
                    m.importance = $importance,
                    m.tags = $tags,
                    m.tag_prefixes = $tag_prefixes,
                    m.type = $type,
                    m.confidence = $confidence,
                    m.t_valid = $t_valid,
                    m.t_invalid = $t_invalid,
                    m.updated_at = $updated_at,
                    m.last_accessed = $last_accessed,
                    m.metadata = $metadata,
                    m.processed = false
                SET m.content = $content,
                    m.timestamp = $timestamp,
                    m.importance = $importance,
                    m.tags = $tags,
                    m.tag_prefixes = $tag_prefixes,
                    m.type = $type,
                    m.confidence = $confidence,
                    m.t_valid = $t_valid,
                    m.t_invalid = $t_invalid,
                    m.updated_at = $updated_at,
                    m.last_accessed = $last_accessed,
                    m.metadata = $metadata,
                    m.processed = false
                RETURN m
                """,
                {
                    "id": memory_id,
                    "content": content,
                    "timestamp": created_at,
                    "importance": importance,
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "type": memory_type,
                    "confidence": type_confidence,
                    "t_valid": t_valid or created_at,
                    "t_invalid": t_invalid,
                    "updated_at": updated_at,
                    "last_accessed": last_accessed,
                    "metadata": metadata_json,
                },
            )
        except Exception:
            logger.exception("Failed to persist memory in FalkorDB")
            abort(500, description="Failed to store memory in FalkorDB")

        if MEMORY_STRICT_VALIDATION:
            _bump_cluster_versions(graph, tags_lower)

        # Queue enrichment
        enqueue_enrichment(memory_id)

        # Handle embeddings
        embedding_status = "skipped"
        qdrant_client = get_qdrant_client()
        if embedding is not None:
            embedding_status = "generated" if embedding_generated else "provided"
            qdrant_result = None
            if qdrant_client is not None:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            point_struct(
                                id=memory_id,
                                vector=embedding,
                                payload={
                                    "content": content,
                                    "tags": tags,
                                    "tag_prefixes": tag_prefixes,
                                    "importance": importance,
                                    "timestamp": created_at,
                                    "type": memory_type,
                                    "confidence": type_confidence,
                                    "t_valid": t_valid or created_at,
                                    "t_invalid": t_invalid,
                                    "updated_at": updated_at,
                                    "last_accessed": last_accessed,
                                    "metadata": metadata,
                                },
                            )
                        ],
                    )
                    qdrant_result = "stored"
                except Exception:
                    logger.exception(
                        "Qdrant upsert failed for memory %s in collection %s",
                        memory_id,
                        collection_name,
                    )
                    qdrant_result = "failed"
        elif qdrant_client is not None:
            enqueue_embedding(memory_id, content)
            embedding_status = "queued"
            qdrant_result = "queued"
        else:
            qdrant_result = "unconfigured"

        response = {
            "status": "success",
            "memory_id": memory_id,
            "stored_at": created_at,
            "type": memory_type,
            "confidence": type_confidence,
            "qdrant": qdrant_result,
            "embedding_status": embedding_status,
            "enrichment": "queued" if state.enrichment_queue else "disabled",
            "metadata": metadata,
            "timestamp": created_at,
            "updated_at": updated_at,
            "last_accessed": last_accessed,
            "query_time_ms": round((time.perf_counter() - query_start) * 1000, 2),
        }

        # Include summarization info in response
        if original_content:
            response["summarized"] = True
            response["original_length"] = len(original_content)
            response["summarized_length"] = len(content)

        if strict_warnings:
            response["warnings"] = strict_warnings
        if duplicate_suspects:
            response["duplicate_suspects"] = duplicate_suspects

        logger.info(
            "memory_stored",
            extra={
                "memory_id": memory_id,
                "type": memory_type,
                "importance": importance,
                "tags_count": len(tags),
                "content_length": len(content),
                "latency_ms": response["query_time_ms"],
                "embedding_status": embedding_status,
                "qdrant_status": qdrant_result,
                "enrichment_queued": bool(state.enrichment_queue),
            },
        )
        return jsonify(response), 201

    @bp.route("/memory/<memory_id>", methods=["GET"])
    def get(memory_id: str) -> ResponseReturnValue:
        _validate_memory_id(memory_id)

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="Graph database unavailable")

        try:
            result = graph.query(
                "MATCH (m:Memory {id: $id}) RETURN m",
                {"id": memory_id},
            )
        except Exception:
            logger.exception("Failed to fetch memory %s", memory_id)
            abort(500, description="Failed to fetch memory")

        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        node = serialize_node(result.result_set[0][0])
        # Parse metadata field for consistency with by_tag endpoint
        node["metadata"] = parse_metadata_field(node.get("metadata"))

        # Update last_accessed timestamp for consistency with by_tag endpoint
        if on_access:
            try:
                on_access([memory_id])
            except Exception:
                logger.exception("on_access failed for memory %s", memory_id)

        return jsonify({"status": "success", "memory": node})

    @bp.route("/memory/<memory_id>", methods=["PATCH"])
    def update(memory_id: str) -> Any:
        _validate_memory_id(memory_id)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        current_node = result.result_set[0][0]
        current = serialize_node(current_node)

        new_content = payload.get("content", current.get("content"))
        tags = normalize_tag_list(payload.get("tags", current.get("tags")))
        tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        tag_prefixes = compute_tag_prefixes(tags_lower)
        importance = payload.get("importance", current.get("importance"))
        memory_type = payload.get("type", current.get("type"))
        confidence = payload.get("confidence", current.get("confidence"))
        timestamp = payload.get("timestamp", current.get("timestamp"))
        t_valid = payload.get("t_valid", current.get("t_valid"))
        t_invalid = payload.get("t_invalid", current.get("t_invalid"))
        metadata_raw = payload.get("metadata", parse_metadata_field(current.get("metadata")))
        updated_at = payload.get("updated_at", current.get("updated_at", utc_now()))
        last_accessed = payload.get("last_accessed", current.get("last_accessed"))

        if metadata_raw is None:
            metadata: Dict[str, Any] = {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            abort(400, description="'metadata' must be an object")
        metadata_json = json.dumps(metadata, default=str)

        if timestamp:
            try:
                timestamp = normalize_timestamp(timestamp)
            except ValueError as exc:
                abort(400, description=f"Invalid timestamp: {exc}")

        if t_valid:
            try:
                t_valid = normalize_timestamp(t_valid)
            except ValueError as exc:
                abort(400, description=f"Invalid t_valid: {exc}")

        if t_invalid:
            try:
                t_invalid = normalize_timestamp(t_invalid)
            except ValueError as exc:
                abort(400, description=f"Invalid t_invalid: {exc}")

        if updated_at:
            try:
                updated_at = normalize_timestamp(updated_at)
            except ValueError as exc:
                abort(400, description=f"Invalid updated_at: {exc}")

        if last_accessed:
            try:
                last_accessed = normalize_timestamp(last_accessed)
            except ValueError as exc:
                abort(400, description=f"Invalid last_accessed: {exc}")

        strict_warnings: List[Dict[str, str]] = []
        if MEMORY_STRICT_VALIDATION:
            # Parity with POST /memory: without these, PATCH bypasses the hard
            # content limit and range coercion entirely.
            if len(new_content or "") > MEMORY_CONTENT_HARD_LIMIT:
                abort(
                    400,
                    description=f"Content exceeds maximum length of {MEMORY_CONTENT_HARD_LIMIT} "
                    f"characters ({len(new_content)} provided).",
                )
            importance = coerce_importance(importance)
            confidence = coerce_importance(confidence)
            # Validate only ADDED tags (supplied minus current): PATCH replaces the
            # tag list wholesale, so a client editing tags MUST resend the node's
            # inherited server-injected entity:* tags — resending what the node
            # already carries is not a client write. Only genuinely new tags are
            # held to the reserved-namespace check.
            current_tags_lower = {
                str(t).strip().lower()
                for t in (current.get("tags") or [])
                if isinstance(t, str) and t.strip()
            }
            client_tags = [
                t
                for t in (
                    normalize_tag_list(payload.get("tags"))
                    if payload.get("tags") is not None
                    else []
                )
                if str(t).strip().lower() not in current_tags_lower
            ]
            memory_type, strict_warnings = _strict_gate(new_content or "", memory_type, client_tags)
            # Renaming onto an existing name is the same duplicate as storing one.
            # Exemption checks the MERGED tags: an existing class:log node stays
            # editable without the client re-sending its tag list.
            if not _is_log_class(tags_lower):
                existing = find_name_collision(graph, new_content or "", exclude_id=memory_id)
                if existing:
                    body = rejection_body(
                        [duplicate_name_finding(memory_name(new_content or ""), existing)],
                        [],
                        authoring_standard(MEMORY_AUTHORING_STANDARD_FILE),
                    )
                    body["existing_memory"] = existing
                    abort(make_response(jsonify(body), 400))

        update_query = """
            MATCH (m:Memory {id: $id})
            SET m.content = $content,
                m.tags = $tags,
                m.tag_prefixes = $tag_prefixes,
                m.importance = $importance,
                m.type = $type,
                m.confidence = $confidence,
                m.timestamp = $timestamp,
                m.t_valid = $t_valid,
                m.t_invalid = $t_invalid,
                m.metadata = $metadata,
                m.updated_at = $updated_at,
                m.last_accessed = $last_accessed
            RETURN m
        """

        graph.query(
            update_query,
            {
                "id": memory_id,
                "content": new_content,
                "tags": tags,
                "tag_prefixes": tag_prefixes,
                "importance": importance,
                "type": memory_type,
                "confidence": confidence,
                "timestamp": timestamp,
                "t_valid": t_valid,
                "t_invalid": t_invalid,
                "metadata": metadata_json,
                "updated_at": updated_at,
                "last_accessed": last_accessed,
            },
        )

        qdrant_client = get_qdrant_client()
        vector = None
        duplicate_suspects: List[Dict[str, Any]] = []
        if qdrant_client is not None:
            if new_content != current.get("content"):
                vector = generate_real_embedding(new_content)
                if MEMORY_STRICT_VALIDATION and not _is_log_class(tags_lower):
                    duplicate_suspects = find_duplicate_suspects(
                        qdrant_client, collection_name, vector, exclude_id=memory_id
                    )
                    if duplicate_suspects:
                        strict_warnings.append(
                            {
                                "check": "duplicate-suspects",
                                "severity": "warn",
                                "message": "Nearest existing memories: "
                                + "; ".join(
                                    f"{s['name']} ({s['id']}, {s['similarity']})"
                                    for s in duplicate_suspects
                                )
                                + " — if one of these is the same fact, update it instead "
                                "of storing a variant.",
                            }
                        )
            else:
                try:
                    existing = qdrant_client.retrieve(
                        collection_name=collection_name,
                        ids=[memory_id],
                        with_vectors=True,
                    )
                    if existing:
                        vector = existing[0].vector
                except Exception:
                    logger.exception("Failed to retrieve existing vector; regenerating")
                    vector = generate_real_embedding(new_content)

            if vector is not None:
                payload = {
                    "content": new_content,
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "importance": importance,
                    "timestamp": timestamp,
                    "type": memory_type,
                    "confidence": confidence,
                    "t_valid": t_valid,
                    "t_invalid": t_invalid,
                    "updated_at": updated_at,
                    "last_accessed": last_accessed,
                    "metadata": metadata,
                }
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[point_struct(id=memory_id, vector=vector, payload=payload)],
                    )
                except Exception:
                    logger.exception(
                        "Qdrant upsert failed for memory %s in collection %s",
                        memory_id,
                        collection_name,
                    )

        update_response: Dict[str, Any] = {"status": "success", "memory_id": memory_id}
        if MEMORY_STRICT_VALIDATION:
            old_tags = [
                t.strip().lower()
                for t in (current.get("tags") or [])
                if isinstance(t, str) and t.strip()
            ]
            _bump_cluster_versions(graph, sorted(set(tags_lower) | set(old_tags)))
            if strict_warnings:
                update_response["warnings"] = strict_warnings
            if duplicate_suspects:
                update_response["duplicate_suspects"] = duplicate_suspects
        return jsonify(update_response)

    @bp.route("/memory/<memory_id>", methods=["DELETE"])
    def delete(memory_id: str) -> Any:
        _validate_memory_id(memory_id)
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
        if not getattr(result, "result_set", None):
            abort(404, description="Memory not found")

        deleted_tags: List[str] = []
        if MEMORY_STRICT_VALIDATION:
            deleted_node = serialize_node(result.result_set[0][0])
            deleted_tags = [
                t.strip().lower()
                for t in (deleted_node.get("tags") or [])
                if isinstance(t, str) and t.strip()
            ]

        graph.query("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})

        if MEMORY_STRICT_VALIDATION:
            _bump_cluster_versions(graph, deleted_tags)

        _delete_qdrant_points(
            qdrant_client=get_qdrant_client(),
            collection_name=collection_name,
            memory_ids=[memory_id],
            logger=logger,
        )

        return jsonify({"status": "success", "memory_id": memory_id})

    @bp.route("/memory/by-tag", methods=["GET", "DELETE"])
    def by_tag() -> Any:
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        tags, limit, offset = _parse_by_tag_request(
            request_args=request.args,
            normalize_tag_list_fn=normalize_tag_list,
            abort_fn=abort,
        )

        if request.method == "DELETE":
            if MEMORY_STRICT_VALIDATION:
                # Strict instances have no bulk removal path at all: a tag match
                # can silently take thousands of memories with no dry-run, which
                # is incompatible with a gate whose premise is deliberate writes.
                abort(
                    403,
                    description="Bulk delete by tag is disabled under strict mode; "
                    "delete memories individually by id.",
                )
            deleted_count = 0
            while True:
                memories, _ = _load_memories_by_tag_page(
                    graph=graph,
                    tags=tags,
                    limit=200,
                    offset=0,
                    serialize_node_fn=serialize_node,
                    parse_metadata_field_fn=parse_metadata_field,
                    logger=logger,
                    abort_fn=abort,
                )
                memory_ids = [str(memory.get("id")) for memory in memories if memory.get("id")]
                if not memory_ids:
                    break

                _delete_graph_memories(
                    graph=graph,
                    memory_ids=memory_ids,
                    logger=logger,
                    abort_fn=abort,
                )

                _delete_qdrant_points(
                    qdrant_client=get_qdrant_client(),
                    collection_name=collection_name,
                    memory_ids=memory_ids,
                    logger=logger,
                )
                deleted_count += len(memory_ids)

            return jsonify({"status": "success", "tags": tags, "deleted_count": deleted_count})

        memories, has_more = _load_memories_by_tag_page(
            graph=graph,
            tags=tags,
            limit=limit,
            offset=offset,
            serialize_node_fn=serialize_node,
            parse_metadata_field_fn=parse_metadata_field,
            logger=logger,
            abort_fn=abort,
        )

        if on_access and memories:
            accessed_ids = [str(m.get("id")) for m in memories if m.get("id")]
            if accessed_ids:
                try:
                    on_access(accessed_ids)
                except Exception:
                    logger.exception("on_access failed for %d memories", len(accessed_ids))

        return jsonify(
            {
                "status": "success",
                "tags": tags,
                "count": len(memories),
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "memories": memories,
            }
        )

    @bp.route("/associate", methods=["POST"])
    def associate() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body is required")
        if isinstance(payload.get("associations"), list):
            return _create_association_batch(
                payload=payload,
                coerce_importance_fn=coerce_importance,
                get_memory_graph_fn=get_memory_graph,
                authorable_relations=set(authorable_relations),
                relationship_types=relation_types,
                utc_now_fn=utc_now,
                abort_fn=abort,
                jsonify_fn=jsonify,
                logger=logger,
            )

        memory1_id = (payload.get("memory1_id") or "").strip()
        memory2_id = (payload.get("memory2_id") or "").strip()
        relation_type = (payload.get("type") or "RELATES_TO").upper()
        strength = coerce_importance(payload.get("strength", 0.5))

        if not memory1_id or not memory2_id:
            abort(400, description="'memory1_id' and 'memory2_id' are required")
        _validate_memory_id(memory1_id)
        _validate_memory_id(memory2_id)
        if memory1_id == memory2_id:
            abort(400, description="Cannot associate a memory with itself")
        if relation_type not in set(authorable_relations):
            abort(
                400,
                description=f"Relation type must be one of {sorted(authorable_relations)}",
            )

        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        timestamp = utc_now()

        relationship_props = _prepare_association_props(
            payload=payload,
            relation_type=relation_type,
            strength=strength,
            timestamp=timestamp,
            relationship_types=relation_types,
        )
        relation_config = relation_types.get(relation_type, {})

        set_clauses = [f"r.{key} = ${key}" for key in relationship_props]
        set_clause = ", ".join(set_clauses)

        try:
            result = graph.query(
                f"""
                MATCH (m1:Memory {{id: $id1}})
                MATCH (m2:Memory {{id: $id2}})
                MERGE (m1)-[r:{relation_type}]->(m2)
                SET {set_clause}
                RETURN r
                """,
                {"id1": memory1_id, "id2": memory2_id, **relationship_props},
            )
        except Exception:
            logger.exception("Failed to create association")
            abort(500, description="Failed to create association")

        if not getattr(result, "result_set", None):
            abort(404, description="One or both memories do not exist")

        response = {
            "status": "success",
            "message": f"Association created between {memory1_id} and {memory2_id}",
            "relation_type": relation_type,
            "strength": strength,
        }
        for prop in relation_config.get("properties", []):
            if prop in relationship_props:
                response[prop] = relationship_props[prop]
        return jsonify(response), 201

    @bp.route("/memory/batch", methods=["POST"])
    def store_batch() -> Any:
        """Store multiple memories in a single request.

        Optimised for benchmark ingestion: batches embedding generation,
        graph writes (UNWIND), and Qdrant upserts into single operations.

        Body: {"memories": [{"content": "...", "tags": [...], ...}, ...]}
        Max 500 memories per request.
        """
        query_start = time.perf_counter()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="JSON body with 'memories' array required")

        memories_input = payload.get("memories", [])
        if not isinstance(memories_input, list) or len(memories_input) == 0:
            abort(400, description="'memories' must be a non-empty array")
        if len(memories_input) > 500:
            abort(400, description="Batch size limit is 500 memories per request")

        # 1. Validate and prepare all memories
        validated = []
        batch_warnings: Dict[str, List[Dict[str, str]]] = {}
        batch_names: Dict[str, int] = {}
        strict_graph = get_memory_graph() if MEMORY_STRICT_VALIDATION else None
        if MEMORY_STRICT_VALIDATION and strict_graph is None:
            abort(503, description="FalkorDB is unavailable")
        for i, mem in enumerate(memories_input):
            if not isinstance(mem, dict):
                abort(400, description=f"Memory at index {i} must be an object")

            content = (mem.get("content") or "").strip()
            if not content:
                abort(400, description=f"Memory at index {i} missing 'content'")
            if len(content) > MEMORY_CONTENT_HARD_LIMIT:
                abort(
                    400,
                    description=f"Memory at index {i} exceeds content limit "
                    f"({len(content)}/{MEMORY_CONTENT_HARD_LIMIT})",
                )

            # Apply soft-limit auto-summarization (same policy as single store)
            content_action = should_summarize_content(
                content, MEMORY_CONTENT_SOFT_LIMIT, MEMORY_CONTENT_HARD_LIMIT
            )
            if (
                content_action == "summarize"
                and MEMORY_AUTO_SUMMARIZE
                and not MEMORY_STRICT_VALIDATION
            ):
                openai_client = get_openai_client() if get_openai_client else None
                if openai_client:
                    summary = summarize_content(
                        content,
                        openai_client,
                        CLASSIFICATION_MODEL,
                        MEMORY_SUMMARY_TARGET_LENGTH,
                    )
                    if summary:
                        logger.info(
                            "Auto-summarized batch memory %d: %d -> %d chars",
                            i,
                            len(content),
                            len(summary),
                        )
                        content = summary

            memory_id = str(uuid.uuid4())
            tags = normalize_tags(mem.get("tags"))
            tags_lower = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
            tag_prefixes = compute_tag_prefixes(tags_lower)
            importance = coerce_importance(mem.get("importance"))
            now = utc_now()
            created_at = now
            if mem.get("timestamp"):
                try:
                    created_at = normalize_timestamp(mem["timestamp"])
                except ValueError:
                    if MEMORY_STRICT_VALIDATION:
                        # Parity with POST /memory, which rejects bad timestamps
                        # instead of silently substituting the current time.
                        abort(
                            400,
                            description=f"Memory at index {i} has an invalid timestamp "
                            f"({mem['timestamp']!r}).",
                        )
                    logger.warning(
                        "Invalid timestamp at index %d (%s), using current time",
                        i,
                        mem["timestamp"],
                    )

            metadata_raw = mem.get("metadata")
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            metadata_json = json.dumps(metadata, default=str)

            memory_type = mem.get("type")
            type_confidence = mem.get("confidence")
            if memory_type:
                type_confidence = (
                    coerce_importance(type_confidence) if type_confidence is not None else 0.9
                )
            else:
                memory_type, type_confidence = memory_classify(content)

            if MEMORY_STRICT_VALIDATION:
                if memory_type:
                    normalized_type, _ = normalize_memory_type(memory_type)
                    if normalized_type:
                        memory_type = normalized_type
                _, findings = validate_memory(
                    content,
                    memory_type,
                    tags,
                    known_types=MEMORY_TYPES,
                    type_aliases=TYPE_ALIASES,
                    contributor_names=MEMORY_STRICT_CONTRIBUTOR_NAMES,
                )
                rejections, item_warnings = split_findings(findings)
                if rejections:
                    body = rejection_body(
                        rejections,
                        item_warnings,
                        authoring_standard(MEMORY_AUTHORING_STANDARD_FILE),
                    )
                    body["index"] = i
                    abort(make_response(jsonify(body), 400))
                if item_warnings:
                    batch_warnings[str(i)] = [w.to_dict() for w in item_warnings]
                # Name-collision gate, batch flavour: also catches two items in
                # THIS batch claiming the same name. No similarity advisory here
                # (bulk ingestion stays single-pass; run the audit scanner after).
                tags_lower_item = [
                    t.strip().lower() for t in tags if isinstance(t, str) and t.strip()
                ]
                if not _is_log_class(tags_lower_item):
                    item_name = memory_name(content)
                    if item_name:
                        prior_index = batch_names.get(item_name)
                        if prior_index is not None:
                            finding = Finding(
                                "duplicate-name",
                                "reject",
                                f"Memories at indexes {prior_index} and {i} both claim the "
                                f"name {item_name!r}; names are identifiers — merge them or "
                                "rename one.",
                            )
                            body = rejection_body(
                                [finding],
                                [],
                                authoring_standard(MEMORY_AUTHORING_STANDARD_FILE),
                            )
                            body["index"] = i
                            abort(make_response(jsonify(body), 400))
                        existing = find_name_collision(strict_graph, content)
                        if existing:
                            body = rejection_body(
                                [duplicate_name_finding(item_name, existing)],
                                [],
                                authoring_standard(MEMORY_AUTHORING_STANDARD_FILE),
                            )
                            body["index"] = i
                            body["existing_memory"] = existing
                            abort(make_response(jsonify(body), 400))
                        batch_names[item_name] = i

            validated.append(
                {
                    "id": memory_id,
                    "content": content,
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "importance": importance,
                    "timestamp": created_at,
                    "type": memory_type,
                    "confidence": type_confidence,
                    "updated_at": created_at,
                    "last_accessed": created_at,
                    "metadata": metadata_json,
                    "metadata_dict": metadata,
                    # t_valid/t_invalid: not accepted in batch endpoint.
                    # Optimised for benchmark ingestion; use POST /memory for full support.
                    "t_valid": created_at,
                    "t_invalid": None,
                }
            )

        # 2. Batch embed all contents
        contents = [v["content"] for v in validated]
        embeddings: List[Optional[List[float]]] = [None] * len(validated)

        if generate_real_embeddings_batch:
            try:
                batch_result = generate_real_embeddings_batch(contents)
                if len(batch_result) != len(contents):
                    logger.warning(
                        "Batch embedding returned %d vectors for %d contents, falling back",
                        len(batch_result),
                        len(contents),
                    )
                    raise ValueError("embedding count mismatch")
                embeddings = batch_result
            except Exception:
                logger.exception("Batch embedding failed, falling back to individual")
                for j, c in enumerate(contents):
                    try:
                        embeddings[j] = generate_real_embedding(c)
                    except Exception as e:
                        logger.warning("Individual embedding failed for item %d: %s", j, e)
        else:
            # Fall back to individual embedding calls
            for j, c in enumerate(contents):
                try:
                    embeddings[j] = generate_real_embedding(c)
                except Exception as e:
                    logger.warning("Individual embedding failed for item %d: %s", j, e)

        # 3. Batch graph write via UNWIND
        graph = get_memory_graph()
        if graph is None:
            abort(503, description="FalkorDB is unavailable")

        try:
            graph.query(
                """
                UNWIND $memories AS m
                MERGE (node:Memory {id: m.id})
                SET
                    node.content = m.content,
                    node.timestamp = m.timestamp,
                    node.importance = m.importance,
                    node.tags = m.tags,
                    node.tag_prefixes = m.tag_prefixes,
                    node.type = m.type,
                    node.confidence = m.confidence,
                    node.t_valid = m.t_valid,
                    node.t_invalid = m.t_invalid,
                    node.updated_at = m.updated_at,
                    node.last_accessed = m.last_accessed,
                    node.metadata = m.metadata,
                    node.processed = false
                RETURN node.id
                """,
                {"memories": validated},
            )
        except Exception:
            logger.exception("Batch graph write failed")
            abort(500, description="Failed to store memories in FalkorDB")

        # 4. Batch Qdrant upsert
        qdrant_client = get_qdrant_client()
        qdrant_status = "unconfigured"
        if qdrant_client is not None:
            points = []
            for v, emb in zip(validated, embeddings, strict=True):
                if emb is not None:
                    points.append(
                        point_struct(
                            id=v["id"],
                            vector=emb,
                            payload={
                                "content": v["content"],
                                "tags": v["tags"],
                                "tag_prefixes": v["tag_prefixes"],
                                "importance": v["importance"],
                                "timestamp": v["timestamp"],
                                "type": v["type"],
                                "confidence": v["confidence"],
                                "updated_at": v["updated_at"],
                                "last_accessed": v["last_accessed"],
                                "metadata": v["metadata_dict"],
                            },
                        )
                    )
            # Enqueue embedding retry for items that failed individually
            failed_emb_ids = [
                v["id"] for v, emb in zip(validated, embeddings, strict=True) if emb is None
            ]
            for fid in failed_emb_ids:
                v_match = next(v for v in validated if v["id"] == fid)
                enqueue_embedding(fid, v_match["content"])

            if points:
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )
                    if failed_emb_ids:
                        qdrant_status = f"stored ({len(points)}), queued ({len(failed_emb_ids)})"
                    else:
                        qdrant_status = f"stored ({len(points)})"
                except Exception:
                    logger.exception("Batch Qdrant upsert failed")
                    # Still queue all embeddings as fallback
                    for v in validated:
                        enqueue_embedding(v["id"], v["content"])
                    qdrant_status = "queued (fallback)"
            else:
                # No embeddings succeeded, queue them all
                for v in validated:
                    enqueue_embedding(v["id"], v["content"])
                qdrant_status = "queued"
        # Queue enrichment for all
        for v in validated:
            enqueue_enrichment(v["id"])

        if MEMORY_STRICT_VALIDATION:
            all_tags = sorted(
                {
                    t.strip().lower()
                    for v in validated
                    for t in (v["tags"] or [])
                    if isinstance(t, str) and t.strip()
                }
            )
            _bump_cluster_versions(graph, all_tags)

        elapsed_ms = round((time.perf_counter() - query_start) * 1000, 2)
        logger.info(
            "batch_stored",
            extra={
                "count": len(validated),
                "latency_ms": elapsed_ms,
                "qdrant_status": qdrant_status,
            },
        )

        batch_response = {
            "status": "success",
            "stored": len(validated),
            "memory_ids": [v["id"] for v in validated],
            "qdrant": qdrant_status,
            "enrichment": "queued" if state.enrichment_queue else "disabled",
            "query_time_ms": elapsed_ms,
        }
        if batch_warnings:
            batch_response["warnings"] = batch_warnings
        return jsonify(batch_response), 201

    return bp
