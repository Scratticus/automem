"""Strict-mode validation tests (Tier 0).

Layout: each group of like tests carries one annotation block —
  TESTS:          what rule the group exercises
  FAILS WHEN:     the malformation that trips it
  PASSES INSTEAD: the shape a correct memory takes

Two layers: unit tests over automem.memory_validation (the checks ported from
the client-side memory-format linter, now pydantic shape models), then route
tests proving the gate at all four write paths — including that DEFAULT
behaviour is untouched when strict mode is off.
"""

import json

import pytest

import app
import automem.api.memory as memory_api
import automem.memory_validation as mv
from automem.config import MEMORY_TYPES, TYPE_ALIASES
from automem.memory_validation import split_findings, validate_memory
from tests.support.fake_graph import FakeGraph

VALID_DECISION = "use-uv-for-python | scope:tooling | tier:2\nWHEN provisioning python\nDO use uv"
VALID_CONTEXT = "project-layout | tier:2\nDEFINES the api lives in automem/api"
VALID_PREFERENCE = "editor-choice | tier:2\nPREFERS neovim OVER vscode"
VALID_INSIGHT = "recall-latency | tier:2\nINSIGHT tag-filtered recall is 10x faster"
VALID_PATTERN = "friday-deploys | tier:2\nWHEN friday afternoon RECURS deploy freezes slip"


def _validate(content, mtype="Decision", tags=(), **kw):
    kw.setdefault("known_types", MEMORY_TYPES)
    kw.setdefault("type_aliases", TYPE_ALIASES)
    return validate_memory(content, mtype, list(tags), **kw)


def _rejects(findings, check):
    return any(f.check == check and f.severity == "reject" for f in findings)


# =============================================================================
# TESTS: every valid shape parses into its model (a valid memory = a parsed one)
# FAILS WHEN: never — these are the positive controls for all seven types
# PASSES INSTEAD: n/a; each constant above is the canonical example of its type
# =============================================================================
class TestValidShapesParse:
    @pytest.mark.parametrize(
        "content,mtype",
        [
            (VALID_DECISION, "Decision"),
            ("inline-form | tier:2\nWHEN closing a letter DO MUST NOT use filler closers", "Style"),
            (VALID_DECISION.replace("use-uv-for-python", "shell-idiom"), "Style"),
            (VALID_DECISION.replace("use-uv-for-python", "morning-review"), "Habit"),
            (VALID_CONTEXT, "Context"),
            ("cluster-hub | tier:1\nANCHOR rules for verification", "Context"),
            (VALID_PREFERENCE, "Preference"),
            (VALID_INSIGHT, "Insight"),
            ("probe-results | tier:2\nEVENT ran probe\nEXPECTED misses\nRESULT hits", "Insight"),
            (VALID_PATTERN, "Pattern"),
        ],
    )
    def test_parses_clean(self, content, mtype):
        model, findings = _validate(content, mtype)
        assert split_findings(findings)[0] == []
        assert model is not None

    def test_transcription_is_render_ready(self):
        # The parsed model doubles as the JSON transcription consumers render from.
        model, _ = _validate(VALID_DECISION, "Decision")
        t = model.transcription()
        assert t == {
            "type": "Decision",
            "name": "use-uv-for-python",
            "tier": 2,
            "scope": "tooling",
            "do": "use uv",
            "when": "provisioning python",
        }


# =============================================================================
# TESTS: per-type body shapes (the authoring standard's <types> section)
# FAILS WHEN: a required body line for the type is absent or malformed
# PASSES INSTEAD: Rule types need "DO {action}"; Context needs DEFINES/ANCHOR;
#   Preference needs "PREFERS {x}"; Insight needs INSIGHT or EVENT+EXPECTED+RESULT;
#   Pattern needs "WHEN {trigger} RECURS {behaviour}"
# =============================================================================
class TestShapeViolationsReject:
    @pytest.mark.parametrize(
        "content,mtype",
        [
            ("no-action | tier:2\nWHEN something happens", "Decision"),
            ("no-action | tier:2\nWHEN something happens", "Style"),
            ("no-action | tier:2\nWHEN something happens", "Habit"),
            ("bad-pref | tier:2\nDO use neovim", "Preference"),
            ("bad-ins | tier:2\nEVENT ran probe\nRESULT hits", "Insight"),
            ("bad-pat | tier:2\nWHEN friday afternoon", "Pattern"),
        ],
    )
    def test_missing_required_line(self, content, mtype):
        _, findings = _validate(content, mtype)
        assert _rejects(findings, f"memory-format/{mtype.lower()}-shape")

    def test_context_needs_defines_or_anchor(self):
        # Context is validated by a model_validator, not a required field.
        _, findings = _validate("bad-ctx | tier:2\nNOTE the api lives somewhere", "Context")
        assert _rejects(findings, "memory-format/context-shape")


# =============================================================================
# TESTS: the top line and memory name (grammar shared by every type)
# FAILS WHEN: first line isn't "{name} | (scope:{x} |) tier:{n}"; or the name
#   contains a type word (uv-decision) — the type field carries that
# PASSES INSTEAD: "use-uv-for-python | scope:tooling | tier:2"
# =============================================================================
class TestTopLine:
    def test_prose_top_line_rejects(self):
        _, findings = _validate("Some prose without a top line.\nDO things")
        assert _rejects(findings, "memory-format/top-line")

    def test_scoped_and_unscoped_both_parse(self):
        for top in ("with-scope | scope:a:b-c | tier:3", "no-scope | tier:3"):
            _, findings = _validate(top + "\nDO things")
            assert not _rejects(findings, "memory-format/top-line")

    def test_name_restating_own_type_rejects(self):
        _, findings = _validate("uv-decision | tier:2\nDO use uv", "Decision")
        assert _rejects(findings, "name-encodes-type")

    def test_other_type_word_in_name_passes(self):
        # "style" as topic vocabulary on a Decision is legitimate (driver-style case)
        _, findings = _validate("driver-style | tier:2\nDO propose one step", "Decision")
        assert not _rejects(findings, "name-encodes-type")

    def test_empty_content_rejects(self):
        _, findings = _validate("")
        assert _rejects(findings, "memory-format/empty")


# =============================================================================
# TESTS: unparsed body lines reject (renderability guarantee)
# FAILS WHEN: a body line starts with a word outside the keyword grammar — a
#   renderer would silently drop it (e.g. the EXCLUDES clause found in the
#   restored corpus, which belongs in a CONTRADICTS-linked scope atom instead)
# PASSES INSTEAD: every body line starts with a grammar keyword
# =============================================================================
class TestUnparsedLines:
    def test_free_prose_line_rejects(self):
        content = VALID_CONTEXT + "\nEXCLUDES chat conversation"
        _, findings = _validate(content, "Context")
        assert _rejects(findings, "memory-format/unparsed-line")

    def test_keyword_only_body_passes(self):
        _, findings = _validate(VALID_CONTEXT + "\nNOTE effective since 0.16", "Context")
        assert not _rejects(findings, "memory-format/unparsed-line")


# =============================================================================
# TESTS: universal checks — type enum, reserved tag namespaces, credentials,
#   contributor attribution (all instance-independent except the name list)
# FAILS WHEN: type unknown to enum+aliases; tag enters the enrichment worker's
#   namespace; content matches an unambiguous credential format; content names
#   a configured contributor (provenance belongs in entity tags, not content)
# PASSES INSTEAD: canonical/alias types; plain tags; secrets referenced by
#   location, never value; attribution-free content
# =============================================================================
class TestUniversalChecks:
    def test_unknown_type_rejects_alias_passes(self):
        _, findings = _validate(VALID_DECISION, "Bananas")
        assert _rejects(findings, "type-enum")
        _, findings = _validate(VALID_DECISION, "decision")
        assert not _rejects(findings, "type-enum")

    @pytest.mark.parametrize("tag", ["entity:tools:uv", "person:jack", "ENTITY:orgs:acme"])
    def test_reserved_tag_rejects(self, tag):
        _, findings = _validate(VALID_DECISION, "Decision", [tag])
        assert _rejects(findings, "reserved-tag-namespace")

    @pytest.mark.parametrize(
        "secret",
        [
            "key AKIAIOSFODNN7EXAMPLE in env",
            "-----BEGIN RSA PRIVATE KEY-----",
            "token ghp_" + "a" * 36,
            "sk-" + "a" * 24,
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload",
        ],
    )
    def test_credential_rejects(self, secret):
        _, findings = _validate(f"leaky | tier:2\nDO note {secret}")
        assert _rejects(findings, "secret-material")

    def test_attribution_only_when_configured(self):
        content = "owner-habit | tier:2\nDO whatever Alice prefers"
        _, findings = _validate(content)
        assert not _rejects(findings, "attribution-in-content")
        _, findings = _validate(content, contributor_names=["Alice"])
        assert _rejects(findings, "attribution-in-content")


# =============================================================================
# TESTS: composition checks — the gate is hard; there is no warn tier
# FAILS WHEN: enumeration markers (e.g./etc — eject examples via EXEMPLIFIES);
#   >1 WHEN or DO (a bundle — split atoms); any WHY line (rationale that changes
#   execution belongs inside the WHEN/DO lines)
# PASSES INSTEAD: single-concept bodies, examples as linked memories, no WHY
# =============================================================================
class TestCompositionRejects:
    def test_enumeration_rejects(self):
        _, findings = _validate("listy | tier:2\nDO avoid tools e.g. hammers")
        assert _rejects(findings, "enumeration-markers")

    def test_bundle_rejects(self):
        _, findings = _validate("bundle | tier:2\nWHEN a DO x\nWHEN b DO y")
        assert _rejects(findings, "atomicity")

    def test_why_rejects(self):
        _, findings = _validate("with-why | tier:2\nDO use uv\nWHY faster")
        assert _rejects(findings, "memory-format/why-clause")


# --- route fixtures ---------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    state = app.ServiceState()
    graph = FakeGraph()
    state.memory_graph = graph
    monkeypatch.setattr(app, "state", state)
    monkeypatch.setattr(app, "init_falkordb", lambda: None)
    monkeypatch.setattr(app, "init_qdrant", lambda: None)
    monkeypatch.setattr(app, "API_TOKEN", "test-token")
    monkeypatch.setattr(app, "ADMIN_TOKEN", "test-admin-token")
    yield graph


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setattr(memory_api, "MEMORY_STRICT_VALIDATION", True)
    monkeypatch.setattr(mv, "_standard_cache", None)
    yield
    monkeypatch.setattr(mv, "_standard_cache", None)


@pytest.fixture
def client():
    return app.app.test_client()


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}


def _post(client, auth_headers, payload, path="/memory"):
    return client.post(
        path, data=json.dumps(payload), content_type="application/json", headers=auth_headers
    )


# =============================================================================
# TESTS: the default (strict OFF) is upstream's permissive behaviour, untouched
# FAILS WHEN: never — these regression-pin the flag-off path for the PR
# PASSES INSTEAD: n/a; free-form content, garbage types, PATCH oversize, and
#   no ClusterVersion writes are all EXPECTED with the flag off
# =============================================================================
class TestDefaultModeUntouched:
    def test_garbage_type_stores(self, client, auth_headers):
        assert (
            _post(client, auth_headers, {"content": "just prose", "type": "Bananas"}).status_code
            == 201
        )

    def test_patch_oversize_passes(self, client, auth_headers):
        memory_id = _post(client, auth_headers, {"content": "small"}).get_json()["memory_id"]
        r = client.patch(
            f"/memory/{memory_id}",
            data=json.dumps({"content": "y" * 2500}),
            content_type="application/json",
            headers=auth_headers,
        )
        assert r.status_code == 200

    def test_no_cluster_versions(self, client, auth_headers, reset_state):
        _post(client, auth_headers, {"content": "plain", "tags": ["t"]})
        assert not any("ClusterVersion" in q for q, _ in reset_state.queries)


# =============================================================================
# TESTS: the strict gate at POST /memory — 400 findings + authoring standard,
#   alias normalization, composition rejections, no silent summarization
#   (the only surviving response warning is the operational size note),
#   ClusterVersion bump on accepted writes
# FAILS WHEN: shape/universal rejections (400 body carries findings + standard);
#   oversized content is stored VERBATIM with a warning, never LLM-rewritten
# PASSES INSTEAD: valid Format-A memories; aliases arrive lowercase and store
#   canonical
# =============================================================================
class TestStoreGate:
    def test_rejection_carries_findings_and_standard(
        self, client, auth_headers, strict, monkeypatch, tmp_path
    ):
        standard = tmp_path / "standard.xml"
        standard.write_text("<standard>write atomic memories</standard>")
        monkeypatch.setattr(memory_api, "MEMORY_AUTHORING_STANDARD_FILE", str(standard))
        r = _post(client, auth_headers, {"content": "prose blob", "type": "Decision"})
        assert r.status_code == 400
        body = r.get_json()
        checks = {f["check"] for f in body["findings"]}
        assert {"memory-format/top-line", "memory-format/decision-shape"} <= checks
        assert body["authoring_standard"].startswith("<standard>")

    def test_valid_store_normalizes_alias_and_bumps_version(
        self, client, auth_headers, strict, reset_state
    ):
        r = _post(
            client,
            auth_headers,
            {"content": VALID_DECISION, "type": "decision", "tags": ["tooling"]},
        )
        assert r.status_code == 201
        assert r.get_json()["type"] == "Decision"
        assert any("ClusterVersion" in q for q, _ in reset_state.queries)

    def test_composition_findings_reject_at_route(self, client, auth_headers, strict):
        content = "warny | tier:2\nDO avoid tools e.g. hammers\nWHY tidier"
        r = _post(client, auth_headers, {"content": content, "type": "Decision"})
        assert r.status_code == 400
        checks = {f["check"] for f in r.get_json()["findings"]}
        assert {"enumeration-markers", "memory-format/why-clause"} <= checks

    def test_never_summarizes(self, client, auth_headers, strict):
        long_do = "DO " + "x" * 600
        r = _post(
            client, auth_headers, {"content": f"long-one | tier:2\n{long_do}", "type": "Decision"}
        )
        assert r.status_code == 201
        body = r.get_json()
        assert "summarized" not in body
        assert any(w["check"] == "content-soft-limit" for w in body["warnings"])


# =============================================================================
# TESTS: strict-mode parity at PATCH — the same limits as POST
# FAILS WHEN: PATCH content exceeds the hard limit or breaks its type's shape
#   (both bypass validation entirely with the flag off — see
#   TestDefaultModeUntouched.test_patch_oversize_passes)
# PASSES INSTEAD: PATCHed content meets the same bar as a fresh store
# =============================================================================
class TestPatchParity:
    def test_inherited_entity_tags_do_not_block_update(self, client, auth_headers, monkeypatch):
        # Enrichment injects entity:* tags server-side; an update that does not
        # touch tags must not be rejected for inheriting them. Store the memory
        # with the gate off (how pre-strict/enriched corpora arose), then update
        # with the gate on.
        stored = _post(
            client,
            auth_headers,
            {"content": VALID_DECISION, "type": "Decision", "tags": ["tooling", "entity:tools:uv"]},
        )
        memory_id = stored.get_json()["memory_id"]
        monkeypatch.setattr(memory_api, "MEMORY_STRICT_VALIDATION", True)
        r = client.patch(
            f"/memory/{memory_id}",
            data=json.dumps({"content": VALID_DECISION.replace("use uv", "always use uv")}),
            content_type="application/json",
            headers=auth_headers,
        )
        assert r.status_code == 200

    def test_client_supplied_reserved_tag_still_rejects_on_update(
        self, client, auth_headers, strict
    ):
        stored = _post(
            client,
            auth_headers,
            {"content": VALID_DECISION, "type": "Decision", "tags": ["tooling"]},
        )
        memory_id = stored.get_json()["memory_id"]
        r = client.patch(
            f"/memory/{memory_id}",
            data=json.dumps({"tags": ["entity:tools:uv"]}),
            content_type="application/json",
            headers=auth_headers,
        )
        assert r.status_code == 400

    def test_oversize_and_bad_shape_reject(self, client, auth_headers, strict):
        stored = _post(
            client,
            auth_headers,
            {"content": VALID_DECISION, "type": "Decision", "tags": ["tooling"]},
        )
        memory_id = stored.get_json()["memory_id"]
        oversize = client.patch(
            f"/memory/{memory_id}",
            data=json.dumps({"content": "y" * 2500}),
            content_type="application/json",
            headers=auth_headers,
        )
        assert oversize.status_code == 400
        bad_shape = client.patch(
            f"/memory/{memory_id}",
            data=json.dumps({"content": "prose again"}),
            content_type="application/json",
            headers=auth_headers,
        )
        assert bad_shape.status_code == 400


# =============================================================================
# TESTS: strict mode at POST /memory/batch — per-index rejection, timestamp
#   parity (single-store 400s on bad timestamps; batch silently substituted
#   "now" — strict makes them equal), warnings keyed by item index
# FAILS WHEN: any item malformed (whole batch 400s, body names the index)
# PASSES INSTEAD: every item valid; advisory items come back under their index
# =============================================================================
class TestBatchGate:
    def test_rejection_names_index(self, client, auth_headers, strict):
        r = _post(
            client,
            auth_headers,
            {
                "memories": [
                    {"content": VALID_DECISION, "type": "Decision"},
                    {"content": "prose", "type": "Decision"},
                ]
            },
            path="/memory/batch",
        )
        assert r.status_code == 400
        assert r.get_json()["index"] == 1

    def test_timestamp_parity(self, client, auth_headers, strict):
        r = _post(
            client,
            auth_headers,
            {
                "memories": [
                    {"content": VALID_DECISION, "type": "Decision", "timestamp": "not-a-date"}
                ]
            },
            path="/memory/batch",
        )
        assert r.status_code == 400

    def test_composition_rejection_names_index(self, client, auth_headers, strict):
        r = _post(
            client,
            auth_headers,
            {
                "memories": [
                    {"content": "warny | tier:2\nDO avoid e.g. hammers", "type": "Decision"}
                ]
            },
            path="/memory/batch",
        )
        assert r.status_code == 400
        assert r.get_json()["index"] == 0


# =============================================================================
# TESTS: ClusterVersion counters — the staleness signal for artifacts rendered
#   from the graph — bump on DELETE too (removing an atom changes its cluster)
# FAILS WHEN: never; asserts the MERGE query runs against the deleted node's tags
# PASSES INSTEAD: n/a
# =============================================================================
class TestBulkDeleteDisabled:
    # TESTS: strict instances have no bulk removal path
    # FAILS WHEN: DELETE /memory/by-tag is attempted with strict mode on (403)
    # PASSES INSTEAD: individual DELETE /memory/<id>, each crossing the delete guard
    def test_by_tag_delete_refused_under_strict(self, client, auth_headers, strict):
        r = client.delete("/memory/by-tag?tags=anything", headers=auth_headers)
        assert r.status_code == 403

    def test_by_tag_delete_allowed_when_off(self, client, auth_headers):
        r = client.delete("/memory/by-tag?tags=anything", headers=auth_headers)
        assert r.status_code != 403


class TestClusterVersions:
    def test_delete_bumps(self, client, auth_headers, strict, reset_state):
        stored = _post(
            client,
            auth_headers,
            {"content": VALID_DECISION, "type": "Decision", "tags": ["tooling"]},
        )
        memory_id = stored.get_json()["memory_id"]
        reset_state.queries.clear()
        assert client.delete(f"/memory/{memory_id}", headers=auth_headers).status_code == 200
        assert any("ClusterVersion" in q for q, _ in reset_state.queries)


# =============================================================================
# TESTS: the duplicate gate — names are identifiers; similarity is advisory
# FAILS WHEN: a store/PATCH/batch item claims an existing memory's name without
#   the class:log record-class tag (400 duplicate-name, carrying the existing
#   memory so the rework is "update it or genuinely rename")
# PASSES INSTEAD: class:log record entries sharing names by design (event logs —
#   e.g. one job-application entry per role on the same day); renames to new
#   names; similar-but-differently-named memories, which come back as ADVISORY
#   duplicate_suspects and warnings, never a rejection (corpus calibration
#   2026-07-07: legitimate siblings can outscore genuine duplicates on cosine)
# =============================================================================

EXISTING = "dup-target | scope:tooling | tier:2\nDO exist first"
COLLIDER = "dup-target | scope:tooling | tier:2\nDO exist twice"
U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
U3 = "00000000-0000-0000-0000-000000000003"


def _seed(graph, mid, content, mtype="Decision", tags=None):
    graph.memories[mid] = {
        "id": mid,
        "content": content,
        "tags": tags or [],
        "type": mtype,
        "importance": 0.5,
        "metadata": "{}",
        "timestamp": "2026-07-07T00:00:00+00:00",
    }


class FakeHit:
    def __init__(self, hit_id, score, content):
        self.id = hit_id
        self.score = score
        self.payload = {"content": content}


class FakeQdrant:
    def __init__(self, hits=()):
        self.hits = list(hits)
        self.upserts = []

    def search(self, **kwargs):
        return self.hits

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def retrieve(self, **kwargs):
        return []


def _patch(client, auth_headers, memory_id, payload):
    return client.patch(
        f"/memory/{memory_id}",
        data=json.dumps(payload),
        content_type="application/json",
        headers=auth_headers,
    )


class TestDuplicateNameGate:
    def test_memory_name_extraction(self):
        assert mv.memory_name(EXISTING) == "dup-target"
        assert mv.memory_name("no top line here") is None
        assert mv.memory_name("") is None

    def test_store_collision_rejects(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        r = _post(client, auth_headers, {"content": COLLIDER, "type": "Decision"})
        assert r.status_code == 400
        body = r.get_json()
        assert any(f["check"] == "duplicate-name" for f in body["findings"])
        assert body["existing_memory"]["id"] == U1
        assert body["existing_memory"]["content"] == EXISTING

    def test_prefix_name_is_not_a_collision(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        r = _post(
            client,
            auth_headers,
            {"content": "dup-target-two | tier:2\nDO differ", "type": "Decision"},
        )
        assert r.status_code == 201

    def test_log_class_incoming_exempt(self, client, auth_headers, strict, reset_state):
        _seed(
            reset_state,
            U1,
            "app-2026-07-07-acme | scope:career | tier:2\nDEFINES applied — Acme, role A.",
            "Context",
            tags=["class:log"],
        )
        r = _post(
            client,
            auth_headers,
            {
                "content": "app-2026-07-07-acme | scope:career | tier:2\n"
                "DEFINES applied — Acme, role B.",
                "type": "Context",
                "tags": ["class:log"],
            },
        )
        assert r.status_code == 201

    def test_log_class_existing_never_blocks(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING, tags=["class:log"])
        r = _post(client, auth_headers, {"content": COLLIDER, "type": "Decision"})
        assert r.status_code == 201

    def test_patch_rename_collision_rejects(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        _seed(reset_state, U2, "other-name | tier:2\nDO differ")
        r = _patch(client, auth_headers, U2, {"content": COLLIDER})
        assert r.status_code == 400
        body = r.get_json()
        assert any(f["check"] == "duplicate-name" for f in body["findings"])
        assert body["existing_memory"]["id"] == U1

    def test_patch_keeps_own_name(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        r = _patch(client, auth_headers, U1, {"content": COLLIDER})
        assert r.status_code == 200

    def test_patch_merged_log_tag_exempts(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        _seed(reset_state, U2, "other-name | tier:2\nDO differ", tags=["class:log"])
        # No tags in the PATCH body: the exemption must come from the MERGED tags.
        r = _patch(client, auth_headers, U2, {"content": COLLIDER})
        assert r.status_code == 200

    def test_validate_reports_collision_without_writing(
        self, client, auth_headers, strict, reset_state
    ):
        _seed(reset_state, U1, EXISTING)
        r = _post(
            client, auth_headers, {"content": COLLIDER, "type": "Decision"}, path="/memory/validate"
        )
        assert r.status_code == 200
        body = r.get_json()
        assert any(f["check"] == "duplicate-name" for f in body["findings"])
        assert body["existing_memory"]["id"] == U1
        assert len(reset_state.memories) == 1  # dry run: nothing stored

    def test_validate_exclude_id_skips_own_node(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        r = _post(
            client,
            auth_headers,
            {"content": COLLIDER, "type": "Decision", "exclude_id": U1},
            path="/memory/validate",
        )
        assert not any(f["check"] == "duplicate-name" for f in r.get_json()["findings"])

    def test_store_suspects_advisory(self, client, auth_headers, strict, reset_state):
        app.state.qdrant = FakeQdrant(
            [
                FakeHit("q-1", 0.93, "near-rule | tier:2\nDO x"),
                FakeHit("q-2", 0.50, "far-rule | tier:2\nDO y"),
            ]
        )
        vec = [0.1] * app.state.effective_vector_size
        r = _post(
            client,
            auth_headers,
            {"content": VALID_DECISION, "type": "Decision", "embedding": vec},
        )
        assert r.status_code == 201
        body = r.get_json()
        assert body["duplicate_suspects"] == [
            {"id": "q-1", "name": "near-rule", "similarity": 0.93}
        ]
        assert any(w["check"] == "duplicate-suspects" for w in body["warnings"])

    def test_log_class_skips_suspects(self, client, auth_headers, strict, reset_state):
        app.state.qdrant = FakeQdrant([FakeHit("q-1", 0.93, "near-rule | tier:2\nDO x")])
        vec = [0.1] * app.state.effective_vector_size
        r = _post(
            client,
            auth_headers,
            {
                "content": "app-2026-07-07-beta | scope:career | tier:2\n"
                "DEFINES applied — Beta GmbH, role C.",
                "type": "Context",
                "tags": ["class:log"],
                "embedding": vec,
            },
        )
        assert r.status_code == 201
        assert "duplicate_suspects" not in r.get_json()

    def test_batch_intra_batch_collision(self, client, auth_headers, strict):
        r = _post(
            client,
            auth_headers,
            {
                "memories": [
                    {"content": EXISTING, "type": "Decision"},
                    {"content": COLLIDER, "type": "Decision"},
                ]
            },
            path="/memory/batch",
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body["index"] == 1
        assert any(f["check"] == "duplicate-name" for f in body["findings"])

    def test_batch_graph_collision(self, client, auth_headers, strict, reset_state):
        _seed(reset_state, U1, EXISTING)
        r = _post(
            client,
            auth_headers,
            {"memories": [{"content": COLLIDER, "type": "Decision"}]},
            path="/memory/batch",
        )
        assert r.status_code == 400
        assert r.get_json()["existing_memory"]["id"] == U1

    def test_batch_log_class_exempt(self, client, auth_headers, strict):
        log_a = "app-2026-07-07-gamma | scope:career | tier:2\n" "DEFINES applied — Gamma, role A."
        log_b = "app-2026-07-07-gamma | scope:career | tier:2\n" "DEFINES skipped — Gamma, role B."
        r = _post(
            client,
            auth_headers,
            {
                "memories": [
                    {"content": log_a, "type": "Context", "tags": ["class:log"]},
                    {"content": log_b, "type": "Context", "tags": ["class:log"]},
                ]
            },
            path="/memory/batch",
        )
        assert r.status_code == 201
        assert r.get_json()["stored"] == 2

    def test_flag_off_collision_stores(self, client, auth_headers, reset_state):
        _seed(reset_state, U1, EXISTING)
        r = _post(client, auth_headers, {"content": COLLIDER, "type": "Decision"})
        assert r.status_code == 201
