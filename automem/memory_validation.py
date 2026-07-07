"""Opt-in strict-mode validation for memory writes (Tier 0).

Strict mode enforces that stored memories stay *machine-renderable*: a memory
that parses cleanly into its type's shape can be compiled into downstream
artifacts (rule packs, skill files, harness prompts) without an LLM pass.
Free-form storage remains the default; strict mode is enabled per instance
via MEMORY_STRICT_VALIDATION=true.

Design: parse, don't validate. Content is parsed into a pydantic model per
shape group (the authoring standard's Rule / State / Observation groups); a
memory is valid exactly when it parses, and the parsed model doubles as the
memory's JSON transcription for consumers that render artifacts from the
graph (skill synthesis, the migration scanner's report).

The gate is hard: every finding rejects the write with HTTP 400. The response
body carries the findings plus the instance's authoring standard
(MEMORY_AUTHORING_STANDARD_FILE), so any client that fails the gate is
simultaneously taught the standard, and the only way through is a rewrite
that satisfies it. There is no acknowledgment or override mechanism.

Everything here is a pure function over (content, type, tags) so the same
code drives the migration scanner without a Flask context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

SEVERITY_REJECT = "reject"
SEVERITY_WARN = "warn"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"check": self.check, "severity": self.severity, "message": self.message}


# ---------------------------------------------------------------------------
# Shape models — one per group in the authoring standard's <types> section.
# A memory is valid iff it parses into its type's model; the parsed model is
# the memory's JSON transcription (see ShapeModel.transcription).
# ---------------------------------------------------------------------------


class ShapeModel(BaseModel):
    """Top-line fields shared by every shape: {name} | (scope:{x} |) tier:{n}."""

    model_config = ConfigDict(extra="ignore")

    type: str
    name: str
    tier: int
    scope: Optional[str] = None

    def transcription(self) -> Dict[str, Any]:
        """The memory as structured JSON — what artifact renderers consume."""
        return self.model_dump(exclude_none=True)


class RuleShape(ShapeModel):
    """Decision / Style / Habit: (WHEN {criteria}) DO {action}."""

    do: str
    when: Optional[str] = None
    exc: Optional[str] = None
    note: Optional[str] = None


class ContextShape(ShapeModel):
    """Context: DEFINES {fact} | ANCHOR {cluster-desc}."""

    defines: Optional[str] = None
    anchor: Optional[str] = None
    note: Optional[str] = None

    @model_validator(mode="after")
    def _defines_or_anchor(self) -> "ContextShape":
        if not (self.defines or self.anchor):
            raise ValueError(
                'Context body must start with "DEFINES {fact}" or "ANCHOR {cluster-desc}".'
            )
        return self


class PreferenceShape(ShapeModel):
    """Preference: PREFERS {x} (OVER {y})."""

    prefers: str


class InsightShape(ShapeModel):
    """Insight: INSIGHT {finding} | EVENT {e} EXPECTED {x} RESULT {r}."""

    insight: Optional[str] = None
    event: Optional[str] = None
    expected: Optional[str] = None
    result: Optional[str] = None

    @model_validator(mode="after")
    def _finding_or_experiment(self) -> "InsightShape":
        if not (self.insight or (self.event and self.expected and self.result)):
            raise ValueError(
                'Insight body must be "INSIGHT {finding}" or EVENT/EXPECTED/RESULT lines.'
            )
        return self


class PatternShape(ShapeModel):
    """Pattern: WHEN {trigger} RECURS {behaviour}."""

    when: str
    recurs: str


SHAPES: Dict[str, Type[ShapeModel]] = {
    "Decision": RuleShape,
    "Style": RuleShape,
    "Habit": RuleShape,
    "Context": ContextShape,
    "Preference": PreferenceShape,
    "Insight": InsightShape,
    "Pattern": PatternShape,
}

# Friendly messages for missing required fields, keyed by (model, field).
_MISSING_FIELD_MESSAGES: Dict[Tuple[Type[ShapeModel], str], str] = {
    (RuleShape, "do"): '{type} (rule) body needs a "DO {{action}}" line.',
    (PreferenceShape, "prefers"): 'Preference body must start with "PREFERS {{x}} (OVER {{y}})".',
    (PatternShape, "when"): 'Pattern body must be "WHEN {{trigger}} RECURS {{behaviour}}".',
    (PatternShape, "recurs"): 'Pattern body must be "WHEN {{trigger}} RECURS {{behaviour}}".',
}


# ---------------------------------------------------------------------------
# Content parsing
# ---------------------------------------------------------------------------

TOP_LINE_RE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9-]*)\s*\|\s*(?:scope:(?P<scope>[a-z0-9:_-]+)\s*\|\s*)?tier:(?P<tier>\d+)\s*$"
)

# Body keywords -> model fields. First occurrence wins; repeats surface as an
# atomicity warning, and unknown line-starts are ignored (matching the linter).
_BODY_KEYWORDS = {
    "WHEN",
    "DO",
    "EXC",
    "NOTE",
    "DEFINES",
    "ANCHOR",
    "PREFERS",
    "INSIGHT",
    "EVENT",
    "EXPECTED",
    "RESULT",
    "RECURS",
}

TYPE_WORDS = {t.lower() for t in SHAPES}


def _parse_body(lines: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, int], List[str]]:
    """-> (fields, keyword counts, unparsed lines). "WHEN x RECURS y" splits into both fields.

    A line whose first word is no keyword cannot carry into the transcription —
    the renderer would silently drop it — so strict mode rejects it (unparsed).
    """
    fields: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    unparsed: List[str] = []
    for line in lines:
        token = line.split(" ", 1)[0].rstrip(":")
        if token not in _BODY_KEYWORDS:
            unparsed.append(line)
            continue
        counts[token] = counts.get(token, 0) + 1
        value = line[len(token) :].lstrip(": ").strip()
        if token == "WHEN" and " RECURS " in value:
            when_part, recurs_part = value.split(" RECURS ", 1)
            fields.setdefault("when", when_part.strip())
            fields.setdefault("recurs", recurs_part.strip())
            counts["RECURS"] = counts.get("RECURS", 0) + 1
            continue
        if token == "WHEN" and " DO " in value:
            # Single-line rule form: "WHEN {criteria} DO {action}"
            when_part, do_part = value.split(" DO ", 1)
            fields.setdefault("when", when_part.strip())
            fields.setdefault("do", do_part.strip())
            counts["DO"] = counts.get("DO", 0) + 1
            continue
        fields.setdefault(token.lower(), value)
    return fields, counts, unparsed


def memory_name(content: str) -> Optional[str]:
    """Extract the identifier from a memory's top line, or None if it has none.

    Names are identifiers: the duplicate gate and the skill renderer both key on
    them, so extraction must agree with TOP_LINE_RE exactly — never a looser split.
    """
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return None
    top = TOP_LINE_RE.match(lines[0])
    return top["name"] if top else None


def parse_memory(
    content: str, memory_type: Optional[str]
) -> Tuple[Optional[ShapeModel], List[Finding]]:
    """Parse content against its type's shape.

    Returns (model, findings). The model is None when the memory doesn't parse
    (or the type has no shape); findings carry every rejection discovered.
    """
    findings: List[Finding] = []
    if not content or not content.strip():
        return None, [Finding("memory-format/empty", SEVERITY_REJECT, "Content is empty.")]

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    top = TOP_LINE_RE.match(lines[0])
    if top:
        name, scope, tier = top["name"], top["scope"], int(top["tier"])
        # A name may reference OTHER type words as topic vocabulary ("driver-style"
        # on a Decision); it must not restate the memory's own type.
        if memory_type and memory_type.lower() in name.split("-"):
            findings.append(
                Finding(
                    "name-encodes-type",
                    SEVERITY_REJECT,
                    f"Memory name restates its own type ({memory_type.lower()}); "
                    "the type field carries that — rename.",
                )
            )
    else:
        name, scope, tier = "", None, 0
        findings.append(
            Finding(
                "memory-format/top-line",
                SEVERITY_REJECT,
                'Top line must be "{name} | (scope:{x} |) tier:{n}" — got ' + repr(lines[0]),
            )
        )

    model_cls = SHAPES.get(memory_type or "")
    if model_cls is None:
        return None, findings  # unknown types are the type-enum check's job

    fields, _, unparsed = _parse_body(lines[1:])
    for line in unparsed:
        if line.split(" ", 1)[0].rstrip(":") == "WHY":
            findings.append(
                Finding(
                    "memory-format/why-clause",
                    SEVERITY_REJECT,
                    "WHY is not in the grammar: rationale that changes execution belongs "
                    "inside the WHEN/DO lines; fold it in or drop it.",
                )
            )
            continue
        findings.append(
            Finding(
                "memory-format/unparsed-line",
                SEVERITY_REJECT,
                f"Body line {line[:60]!r} starts with no known keyword — a renderer would "
                "silently drop it. Rework it into the type's shape, or split it into its "
                "own memory (e.g. a scope boundary becomes a CONTRADICTS-linked atom).",
            )
        )
    try:
        model = model_cls(type=memory_type, name=name, scope=scope, tier=tier, **fields)
    except ValidationError as exc:
        check = f"memory-format/{(memory_type or 'unknown').lower()}-shape"
        for err in exc.errors():
            if err["type"] == "missing":
                template = _MISSING_FIELD_MESSAGES.get((model_cls, str(err["loc"][0])), "")
                message = (
                    template.format(type=memory_type)
                    if template
                    else (f"{memory_type} body is missing its {err['loc'][0].upper()} line.")
                )
            else:
                message = err["msg"].removeprefix("Value error, ")
            findings.append(Finding(check, SEVERITY_REJECT, message))
        return None, findings

    return (model if not findings else None), findings


# ---------------------------------------------------------------------------
# Universal checks (instance-independent: properties of the system itself)
# ---------------------------------------------------------------------------

# Tag namespaces owned by the enrichment worker; client writes would collide
# with server-injected tags and corrupt provenance.
RESERVED_TAG_PREFIXES: Sequence[str] = ("entity:", "person:")

# High-signal credential formats only. Deliberately narrow: a false positive
# blocks a legitimate memory, so each pattern must be unambiguous.
SECRET_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key-block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("openai-style-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("jwt-bearer", re.compile(r"\bBearer eyJ[A-Za-z0-9_-]{10,}\.")),
)


def check_type(
    memory_type: Optional[str], known_types: Iterable[str], type_aliases: Iterable[str]
) -> List[Finding]:
    if not memory_type or memory_type in set(known_types) or memory_type in set(type_aliases):
        return []
    return [
        Finding(
            "type-enum",
            SEVERITY_REJECT,
            f"Unknown memory type {memory_type!r}. Valid types: "
            f"{', '.join(sorted(known_types))} (or a documented alias).",
        )
    ]


def check_tags(tags: Sequence[str]) -> List[Finding]:
    return [
        Finding(
            "reserved-tag-namespace",
            SEVERITY_REJECT,
            f"Tag {tag!r} uses a server-owned namespace ({', '.join(RESERVED_TAG_PREFIXES)}) "
            "injected by the enrichment worker; remove it from the write.",
        )
        for tag in tags or []
        if isinstance(tag, str) and tag.lower().startswith(tuple(RESERVED_TAG_PREFIXES))
    ]


def check_secrets(content: str) -> List[Finding]:
    return [
        Finding(
            "secret-material",
            SEVERITY_REJECT,
            f"Content matches a credential format ({name}). Secrets must never be stored; "
            "reference the secret's location instead.",
        )
        for name, pattern in SECRET_PATTERNS
        if content and pattern.search(content)
    ]


def check_attribution(content: str, contributor_names: Sequence[str]) -> List[Finding]:
    """Provenance lives in tags/metadata (enrichment records contributors);
    content itself stays attribution-free. Names come from instance config."""
    return [
        Finding(
            "attribution-in-content",
            SEVERITY_REJECT,
            f"Content names a contributor ({name}); provenance belongs to entity tags "
            "and metadata, not content — remove the name.",
        )
        for name in contributor_names
        if name and re.search(rf"\b{re.escape(name)}('s|s')?\b", content or "")
    ]


# ---------------------------------------------------------------------------
# Composition checks (reject: single-concept discipline is part of the gate)
# ---------------------------------------------------------------------------

_ENUMERATION_RE = re.compile(r"\be\.g\.|\bi\.e\.|\betc\.?\b|for example|such as", re.I)


def composition_findings(content: str) -> List[Finding]:
    findings: List[Finding] = []
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    _, counts, _ = _parse_body(lines[1:])

    if _ENUMERATION_RE.search(content or ""):
        findings.append(
            Finding(
                "enumeration-markers",
                SEVERITY_REJECT,
                "Content reads like an example/enumeration — cut it, or eject examples "
                "to their own memories linked via EXEMPLIFIES.",
            )
        )
    if counts.get("WHEN", 0) > 1 or counts.get("DO", 0) > 1:
        findings.append(
            Finding(
                "atomicity",
                SEVERITY_REJECT,
                f"{counts.get('WHEN', 0)} WHEN / {counts.get('DO', 0)} DO lines — likely "
                "more than one concept; split into single-concept memories.",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def validate_memory(
    content: str,
    memory_type: Optional[str],
    tags: Sequence[str],
    *,
    known_types: Iterable[str],
    type_aliases: Iterable[str],
    contributor_names: Sequence[str] = (),
) -> Tuple[Optional[ShapeModel], List[Finding]]:
    """Run all strict-mode checks.

    Returns (parsed model | None, findings). Pure function — also drives the
    migration scanner, whose report rows are the models' transcriptions.
    """
    findings = check_type(memory_type, known_types, type_aliases)
    findings += check_tags(tags)
    findings += check_secrets(content)
    findings += check_attribution(content, contributor_names)
    model, shape_findings = parse_memory(content, memory_type)
    findings += shape_findings
    findings += composition_findings(content)
    return model, findings


def split_findings(findings: Sequence[Finding]) -> Tuple[List[Finding], List[Finding]]:
    """-> (rejections, warnings)"""
    rejections = [f for f in findings if f.severity == SEVERITY_REJECT]
    warnings = [f for f in findings if f.severity == SEVERITY_WARN]
    return rejections, warnings


_standard_cache: Optional[str] = None


def authoring_standard(path: Optional[str]) -> str:
    """The instance's authoring standard, attached to every 400 body. Cached."""
    global _standard_cache
    if _standard_cache is None:
        text = ""
        if path:
            try:
                with open(path) as fh:
                    text = fh.read().strip()
            except OSError:
                text = ""
        _standard_cache = text
    return _standard_cache


def rejection_body(
    rejections: Sequence[Finding], warnings: Sequence[Finding], standard: str
) -> Dict[str, Any]:
    """JSON body for a strict-mode 400: findings + the standard that teaches the fix."""
    body: Dict[str, Any] = {
        "error": "strict-mode validation failed",
        "findings": [f.to_dict() for f in list(rejections) + list(warnings)],
    }
    if standard:
        body["authoring_standard"] = standard
    return body
