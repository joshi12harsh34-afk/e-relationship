"""Minimal Semantica entity / relation / triplet extraction.

Self-contained under entity-relationship/ — no imports or paths from the
parent repo. Only pip packages (see requirements.txt) are external.

Layout (this folder only):
  .env                               — Bedrock credentials
  policies/identity_resolution.yaml  — NER taxonomy + predicate hints
  prompts/unified_extraction.txt     — domain guidance for the LLM
  inputs/*.md                        — source documents
  outputs/                           — JSON results (created on run)

Pipeline sections in this file (top → bottom = run order):
  1) Settings          2) Policy             3) Prompts
  4) Bedrock           5) Shared helpers     6) Document prep (boilerplate)
  7) Evidence          8) Typed merge        9) Guided Semantica hooks
  10) Identity/dedupe  11) Graph validation  12) FactExtractor
  13) CLI

Run from this folder:
  python main.py
  python main.py --file inputs/commercial.md
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Optional, Type

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from semantica.deduplication import DuplicateDetector, EntityMerger
from semantica.semantic_extract import (
    CoreferenceResolver,
    EventDetector,
    NamedEntityRecognizer,
    RelationExtractor,
    TripletExtractor,
)
from semantica.semantic_extract.providers import BaseProvider
from semantica.semantic_extract.registry import MethodRegistry, provider_registry
from semantica.split import list_available_methods as list_split_methods, split_recursive
from semantica.utils.exceptions import ProcessingError

# ---------------------------------------------------------------------------
# Paths — everything resolves relative to this folder
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / 'policies' / 'identity_resolution.yaml'
PROMPT_DIR = ROOT / 'prompts'
INPUT_DIR = ROOT / 'inputs'
OUTPUT_DIR = ROOT / 'outputs'

logger = logging.getLogger('entity_relationship')

# ---------------------------------------------------------------------------
# 1) Settings — Bedrock credentials from this folder's .env only
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )
    aws_region: str = 'us-east-1'
    bedrock_region: str = ''
    bedrock_model_id: str = 'us.amazon.nova-2-lite-v1:0'
    bedrock_api_key: str = ''
    bedrock_max_tokens: int = 4096
    bedrock_temperature: float = 0.0
    llm_max_retries: int = 3
    min_extraction_confidence: float = 0.35
    extraction_max_tokens: int = 8192
    # Source chars per Semantica chunk. Keep modest: unified prompt overhead is large.
    extraction_chunk_chars: int = 12000

    @property
    def region(self) -> str:
        return self.bedrock_region or self.aws_region


def get_settings() -> Settings:
    return Settings()

# ---------------------------------------------------------------------------
# 2) Policy — load identity_resolution.yaml (taxonomy + predicate hints)
# ---------------------------------------------------------------------------
class Policy:
    def __init__(self, data: dict, path: Path | None = None):
        self.data = data or {}
        self.path = path

    def list_values(self, *keys, default: list | None = None) -> list:
        current: Any = self.data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return list(default or [])
            current = current[key]
        if current is None:
            return list(default or [])
        return list(current) if isinstance(current, list) else [current]


def load_policy(path: Path | None = None) -> Policy:
    target = path or POLICY_PATH
    if not target.is_file():
        return Policy({}, path=target)
    with target.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    return Policy(data if isinstance(data, dict) else {}, path=target)


def primary_type_hints(policy: Policy) -> list[str]:
    """TYPE_HINTS from ner_taxonomy.entity_types."""
    taxonomy = policy.data.get('ner_taxonomy') or {}
    out: list[str] = []
    seen: set[str] = set()
    for item in taxonomy.get('entity_types') or []:
        if not isinstance(item, dict):
            continue
        label = collapse(str(item.get('type') or item.get('label') or ''))
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _policy_labels(policy: Policy, section: str, key: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in policy.list_values(section, key):
        label = collapse(str(item or ''))
        fold = label.casefold()
        if not label or fold in seen:
            continue
        seen.add(fold)
        out.append(label)
    return out


def relation_predicate_hints(policy: Policy) -> list[str]:
    hints: list[str] = []
    for key in (
        'hierarchy_predicate_markers',
        'technical_predicate_markers',
        'compliance_predicate_markers',
        'document_reference_predicate_markers',
        'location_predicate_markers',
    ):
        hints.extend(_policy_labels(policy, 'hierarchy', key))
    return list(dict.fromkeys(hints))


def attribute_predicate_hints(policy: Policy) -> list[str]:
    hints = _policy_labels(policy, 'hierarchy', 'attribute_predicate_markers')
    hints.extend(_policy_labels(policy, 'hierarchy', 'attribute_hierarchy_predicates'))
    return list(dict.fromkeys(hints))


def non_entity_type_hints(policy: Policy) -> list[str]:
    return _policy_labels(policy, 'hierarchy', 'attribute_hierarchy_object_types')


def _first_sentence(text: str, *, limit: int = 120) -> str:
    raw = collapse(text or '')
    if not raw:
        return ''
    for sep in ('. ', '; ', '\n'):
        if sep in raw:
            raw = raw.split(sep, 1)[0].rstrip('.;')
            break
    if len(raw) > limit:
        return raw[: max(0, limit - 1)].rstrip() + '…'
    return raw


def entity_definition_hints(policy: Policy, *, limit: int = 28) -> str:
    """Compact type definitions from ner_taxonomy.entity_types for prompts."""
    taxonomy = policy.data.get('ner_taxonomy') if isinstance(policy.data, dict) else {}
    if not isinstance(taxonomy, dict):
        return ''
    lines: list[str] = []
    for item in taxonomy.get('entity_types') or []:
        if not isinstance(item, dict):
            continue
        label = collapse(str(item.get('type') or item.get('label') or ''))
        definition = _first_sentence(str(item.get('definition') or ''), limit=100)
        if not label or not definition:
            continue
        lines.append(f'- {label}: {definition}')
        if len(lines) >= limit:
            break
    return '\n'.join(lines)


def compact_prompt_settings(policy: Policy) -> dict[str, Any]:
    """Load ner_taxonomy.compact_prompt (YAML-driven large-doc / chunked NER hints)."""
    taxonomy = policy.data.get('ner_taxonomy') if isinstance(policy.data, dict) else {}
    if not isinstance(taxonomy, dict):
        return {}
    raw = taxonomy.get('compact_prompt')
    return raw if isinstance(raw, dict) else {}


def compact_priority_subtypes(policy: Policy) -> dict[str, list[str]]:
    """Primary → preferred subtype order for compact prompts (from YAML)."""
    raw = compact_prompt_settings(policy).get('priority_subtypes')
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for primary, values in raw.items():
        key = collapse(str(primary or ''))
        if not key or not isinstance(values, list):
            continue
        ordered = [collapse(str(v or '')) for v in values if collapse(str(v or ''))]
        if ordered:
            out[key] = ordered
    return out


def compact_prompt_pins(policy: Policy) -> list[str]:
    """Short YAML pins appended to compact ontology hints."""
    pins: list[str] = []
    for item in compact_prompt_settings(policy).get('pins') or []:
        text = collapse(str(item or ''))
        if text:
            pins.append(text)
    return pins


def disambiguation_hints(
    policy: Policy,
    *,
    limit: int = 24,
    prioritize: bool = False,
) -> str:
    """Disambiguation rules from ner_taxonomy.disambiguation_rules.

    When prioritize=True (compact / large-doc prompts), rules with
    ``compact_priority`` on the YAML rule surface first (lower number = earlier).
    Rules without the field keep YAML list order after prioritized ones.
    """
    taxonomy = policy.data.get('ner_taxonomy') if isinstance(policy.data, dict) else {}
    if not isinstance(taxonomy, dict):
        return ''
    rows: list[tuple[int, int, str]] = []
    for index, item in enumerate(taxonomy.get('disambiguation_rules') or []):
        if not isinstance(item, dict):
            continue
        between = [
            collapse(str(x or ''))
            for x in (item.get('between') or [])
            if collapse(str(x or ''))
        ]
        rule = _first_sentence(str(item.get('rule') or ''), limit=160)
        if not between or not rule:
            continue
        if prioritize:
            raw_pri = item.get('compact_priority')
            try:
                priority = int(raw_pri) if raw_pri is not None else 10_000
            except (TypeError, ValueError):
                priority = 10_000
            if priority < 0:
                priority = 10_000
        else:
            priority = index
        rows.append((priority, index, f'- {" vs ".join(between)}: {rule}'))
    rows.sort(key=lambda row: (row[0], row[1]))
    return '\n'.join(line for _p, _i, line in rows[:limit])


def recommended_subtype_hints(
    policy: Policy,
    *,
    per_type: int = 8,
    priority_subtypes: dict[str, list[str]] | None = None,
) -> str:
    """Recommended subtypes from ner_taxonomy.recommended_subtypes."""
    taxonomy = policy.data.get('ner_taxonomy') if isinstance(policy.data, dict) else {}
    if not isinstance(taxonomy, dict):
        return ''
    mapping = taxonomy.get('recommended_subtypes') or {}
    if not isinstance(mapping, dict):
        return ''
    lines: list[str] = []
    for label, values in mapping.items():
        primary = collapse(str(label or ''))
        if not primary or not isinstance(values, list):
            continue
        subtypes = [
            collapse(str(v or ''))
            for v in values
            if collapse(str(v or ''))
        ]
        preferred = [
            collapse(str(v or ''))
            for v in (priority_subtypes or {}).get(primary, [])
            if collapse(str(v or ''))
        ]
        if preferred:
            seen: set[str] = set()
            ordered: list[str] = []
            for name in preferred + subtypes:
                key = name.casefold()
                if not name or key in seen:
                    continue
                if name not in subtypes and name not in preferred:
                    continue
                # Keep only ontology-listed subtypes (preferred may re-order).
                if name.casefold() not in {s.casefold() for s in subtypes}:
                    continue
                seen.add(key)
                ordered.append(name)
            subtypes = ordered
        subtypes = subtypes[:per_type]
        if subtypes:
            lines.append(f'- {primary}: {", ".join(subtypes)}')
    return '\n'.join(lines)


def table_extraction_hints(policy: Policy) -> str:
    """Table column-role / formatting guidance from table_extraction YAML."""
    table = policy.data.get('table_extraction') if isinstance(policy.data, dict) else {}
    if not isinstance(table, dict):
        return ''
    parts: list[str] = []
    for key, title in (
        ('subject_column_markers', 'Subject/tag columns'),
        ('description_column_markers', 'Description columns'),
        ('quantity_column_markers', 'Quantity columns'),
        ('measurement_column_markers', 'Measurement columns'),
    ):
        markers = [
            collapse(str(v or ''))
            for v in (table.get(key) or [])
            if collapse(str(v or ''))
        ]
        if markers:
            parts.append(f'{title}: {", ".join(markers[:12])}')
    if table.get('keep_verbatim_table_in_prompt'):
        parts.append('Keep verbatim table cells; headers are context, not entities.')
    if table.get('format_row_groups_for_llm'):
        parts.append('Interpret row groups using section/table headers as parent context.')
    return '\n'.join(f'- {p}' for p in parts)


# Section titles for ner_taxonomy.non_entity_policy (definition-based LLM guidance).
_NON_ENTITY_POLICY_SECTIONS: tuple[tuple[str, str], ...] = (
    ('independent_entity', 'Independent entity'),
    ('properties_not_nodes', 'Properties are not entities'),
    ('identifiers_not_separate_nodes', 'Identifiers are not separate entities'),
    ('relations_not_nodes', 'Relations are not entities'),
    ('reject_noise', 'Noise and formatting artifacts'),
    ('reject_field_labels', 'Field labels are not entities'),
    ('reject_generic_words', 'Generic terms (context-sensitive)'),
    ('reject_incomplete_headings', 'Incomplete headings'),
    ('reject_duplicate_identifiers', 'Duplicate identifiers'),
    ('context_sensitive', 'Context-sensitive spans'),
    ('other_is_not_for', 'Other is not for'),
)


def _non_entity_section_payload(raw: Any) -> tuple[str, list[str]]:
    """Normalize a non_entity_policy section to (definition, decision_criteria).

    Supports the definition-based dict shape and legacy list-of-strings shape so
    older policy snippets keep rendering instead of silently dropping guidance.
    """
    if isinstance(raw, dict):
        definition = collapse(str(raw.get('definition') or ''))
        criteria_raw = raw.get('decision_criteria')
        if criteria_raw is None:
            # Backward-compatible: allow legacy keys that still hold string lists.
            for legacy_key in ('criteria', 'examples', 'items'):
                if isinstance(raw.get(legacy_key), list):
                    criteria_raw = raw.get(legacy_key)
                    break
        criteria = [
            collapse(str(item or ''))
            for item in (criteria_raw or [])
            if collapse(str(item or ''))
        ]
        return definition, criteria
    if isinstance(raw, list):
        return '', [collapse(str(item or '')) for item in raw if collapse(str(item or ''))]
    if isinstance(raw, str) and collapse(raw):
        return collapse(raw), []
    return '', []


def non_entity_policy_section(policy: Policy) -> dict[str, Any]:
    """Return ner_taxonomy.non_entity_policy dict (empty if absent)."""
    taxonomy = policy.data.get('ner_taxonomy') if isinstance(policy.data, dict) else {}
    if not isinstance(taxonomy, dict):
        return {}
    section = taxonomy.get('non_entity_policy')
    return section if isinstance(section, dict) else {}


def non_entity_policy_hints(policy: Policy, *, compact: bool = False) -> str:
    """Format non_entity_policy definitions for LLM prompts (no static word lists)."""
    nep = non_entity_policy_section(policy)
    if not nep:
        return ''

    def_limit = 110 if compact else 180
    criteria_limit = 1 if compact else 4
    parts: list[str] = [
        'Non-entity policy (classify from definition, independent identity, and '
        'source evidence — never from memorised example phrases or blacklist words):',
    ]
    for key, title in _NON_ENTITY_POLICY_SECTIONS:
        if key not in nep:
            continue
        definition, criteria = _non_entity_section_payload(nep.get(key))
        if not definition and not criteria:
            continue
        if definition:
            parts.append(f'- {title}: {_first_sentence(definition, limit=def_limit)}')
        else:
            parts.append(f'- {title}:')
        for criterion in criteria[:criteria_limit]:
            snippet = _first_sentence(criterion, limit=120 if compact else 160)
            if snippet:
                parts.append(f'  - {snippet}')
    return '\n'.join(parts) if len(parts) > 1 else ''


def ontology_hints(policy: Policy, *, compact: bool = False) -> str:
    """Ontology/hierarchy/typing hints derived from identity_resolution.yaml."""
    taxonomy = policy.data.get('ner_taxonomy') if isinstance(policy.data, dict) else {}
    hierarchy = policy.data.get('hierarchy') if isinstance(policy.data, dict) else {}
    if not isinstance(taxonomy, dict):
        taxonomy = {}
    if not isinstance(hierarchy, dict):
        hierarchy = {}

    parts: list[str] = []
    schema = collapse(str(taxonomy.get('schema_name') or ''))
    version = collapse(str(taxonomy.get('version') or ''))
    if schema:
        parts.append(f'Taxonomy: {schema}' + (f' v{version}' if version else ''))
    purpose = collapse(str(taxonomy.get('purpose') or ''))
    if purpose:
        parts.append(purpose[:240])

    definitions = entity_definition_hints(policy, limit=16 if compact else 28)
    if definitions:
        parts.append('Entity definitions (classify from source role, not name shape):')
        parts.append(definitions)

    disambiguation = disambiguation_hints(
        policy,
        limit=12 if compact else 24,
        prioritize=compact,
    )
    if disambiguation:
        parts.append('Disambiguation:')
        parts.append(disambiguation)

    subtypes = recommended_subtype_hints(
        policy,
        per_type=6 if compact else 10,
        priority_subtypes=compact_priority_subtypes(policy) if compact else None,
    )
    if subtypes:
        parts.append('Recommended subtypes (use when source supports):')
        parts.append(subtypes)

    if compact:
        for pin in compact_prompt_pins(policy):
            parts.append(pin)

    non_entity = non_entity_policy_hints(policy, compact=compact)
    if non_entity:
        parts.append(non_entity)

    table_hints = table_extraction_hints(policy)
    if table_hints:
        parts.append('Table extraction rules:')
        parts.append(table_hints)

    pairs: list[str] = []
    for item in hierarchy.get('allowed_parent_child') or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parent = collapse(str(item[0] or ''))
            child = collapse(str(item[1] or ''))
            if parent and child:
                pairs.append(f'{parent}->{child}')
    if pairs:
        parts.append('Allowed parent->child: ' + ', '.join(pairs[:28]))

    flags: list[str] = []
    for key in (
        'forbid_self_parent',
        'prevent_cycles',
        'enforce_allowed_parent_child',
        'require_grounded_parent_child_assertion',
        'allow_multiple_supported_parents',
    ):
        if key in hierarchy:
            flags.append(f'{key}={hierarchy.get(key)}')
    if flags:
        parts.append('Hierarchy rules: ' + '; '.join(flags))

    return '\n'.join(parts) if parts else '(none)'


def allowed_parent_child_pairs(policy: Policy) -> set[tuple[str, str]]:
    hierarchy = policy.data.get('hierarchy') if isinstance(policy.data, dict) else {}
    if not isinstance(hierarchy, dict):
        return set()
    pairs: set[tuple[str, str]] = set()
    for item in hierarchy.get('allowed_parent_child') or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parent = collapse(str(item[0] or '')).casefold()
            child = collapse(str(item[1] or '')).casefold()
            if parent and child:
                pairs.add((parent, child))
    return pairs


def hierarchy_predicate_set(policy: Policy) -> set[str]:
    return {p.casefold() for p in _policy_labels(policy, 'hierarchy', 'hierarchy_predicate_markers')}


def predicate_subtype_gates(policy: Policy) -> dict[str, dict[str, Any]]:
    """Load hierarchy.predicate_parent_child_subtypes from policy YAML."""
    hierarchy = policy.data.get('hierarchy') if isinstance(policy.data, dict) else {}
    if not isinstance(hierarchy, dict):
        return {}
    raw = hierarchy.get('predicate_parent_child_subtypes') or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for predicate, gate in raw.items():
        key = collapse(str(predicate or '')).casefold()
        if not key or not isinstance(gate, dict):
            continue
        out[key] = gate
    return out


def _subtype_gate_violation(
    gate: dict[str, Any],
    *,
    parent_subtype: str,
    child_subtype: str,
) -> str | None:
    """Return warning text when parent/child subtypes violate a YAML gate; None if ok."""
    parent = collapse(parent_subtype or '').casefold()
    child = collapse(child_subtype or '').casefold()
    allow_unspecified_parent = bool(gate.get('allow_unspecified_parent_subtype', True))
    allow_unspecified_child = bool(gate.get('allow_unspecified_child_subtype', True))

    parent_allow = {
        collapse(str(v or '')).casefold()
        for v in (gate.get('parent') or [])
        if collapse(str(v or ''))
    }
    child_allow = {
        collapse(str(v or '')).casefold()
        for v in (gate.get('child') or [])
        if collapse(str(v or ''))
    }
    parent_exclude = {
        collapse(str(v or '')).casefold()
        for v in (gate.get('parent_exclude') or [])
        if collapse(str(v or ''))
    }
    child_exclude = {
        collapse(str(v or '')).casefold()
        for v in (gate.get('child_exclude') or [])
        if collapse(str(v or ''))
    }

    if parent_allow:
        if not parent:
            if not allow_unspecified_parent:
                return 'missing required parent subtype'
        elif parent not in parent_allow:
            return f'parent subtype {parent_subtype!r} not in allowed {sorted(parent_allow)[:6]}'
    if parent and parent in parent_exclude:
        return f'parent subtype {parent_subtype!r} is excluded'

    if child_allow:
        if not child:
            if not allow_unspecified_child:
                return 'missing required child subtype'
        elif child not in child_allow:
            return f'child subtype {child_subtype!r} not in allowed {sorted(child_allow)[:6]}'
    if child and child in child_exclude:
        return f'child subtype {child_subtype!r} is excluded'
    return None

# ---------------------------------------------------------------------------
# 3) Prompts — load unified_extraction.txt and fill <<TOKENS>>
# ---------------------------------------------------------------------------
def load_prompt(directory: Path, name: str, **tokens: str) -> str:
    path = directory / f'{name}.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Prompt template not found: {path}')
    text = path.read_text(encoding='utf-8')
    for key, value in tokens.items():
        text = text.replace(f'<<{key}>>', value)
    return text

# ---------------------------------------------------------------------------
# 4) Bedrock — Semantica-compatible chat provider
# ---------------------------------------------------------------------------
class BedrockProvider(BaseProvider):
    """Chat provider backed by Amazon Bedrock Converse API."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        settings = get_settings()
        self.model = model or settings.bedrock_model_id
        self.region = region or settings.region
        self.api_key = api_key or settings.bedrock_api_key or os.environ.get('AWS_BEARER_TOKEN_BEDROCK', '')
        self.max_tokens = int(kwargs.get('max_tokens') or settings.bedrock_max_tokens)
        self.temperature = float(kwargs.get('temperature') if kwargs.get('temperature') is not None else settings.bedrock_temperature)
        self.max_retries = int(kwargs.get('max_retries') or settings.llm_max_retries)
        if self.api_key:
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = self.api_key
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.region,
            config=Config(max_pool_connections=50, retries={'max_attempts': 3, 'mode': 'standard'}, read_timeout=120, connect_timeout=30),
        )

    def is_available(self) -> bool:
        return bool(self.model and self.client)

    def generate(self, prompt: str, **kwargs) -> str:
        model_id = kwargs.get('model') or self.model
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = int(kwargs.get('max_tokens') or self.max_tokens)
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.converse(
                    modelId=model_id,
                    messages=[{'role': 'user', 'content': [{'text': prompt}]}],
                    inferenceConfig={'maxTokens': max_tokens, 'temperature': float(temperature or 0.0)},
                )
                blocks = response['output']['message']['content']
                text = '\n'.join(block['text'] for block in blocks if isinstance(block, dict) and block.get('text'))
                if not text.strip():
                    raise ProcessingError('Bedrock returned an empty response')
                return text
            except (ClientError, BotoCoreError, KeyError, ProcessingError) as exc:
                last_error = exc
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(min(16, 2 ** attempt))
        raise ProcessingError(f'Bedrock generation failed: {last_error}')

    def generate_typed(self, prompt: str, schema: Type[BaseModel], max_retries: int = 3, **kwargs) -> BaseModel:
        retries = max(1, int(max_retries or self.max_retries))
        last_error: Exception | None = None
        requested = int(kwargs.get('max_tokens') or self.max_tokens)
        call_kwargs = {k: v for k, v in kwargs.items() if k != 'max_tokens'}
        for attempt in range(retries):
            try:
                attempt_tokens = min(8192, requested + attempt * 1024)
                if getattr(schema, '__name__', '') == 'EntitiesResponse':
                    (ROOT / 'debug_entity_prompt.txt').write_text(prompt, encoding='utf-8')
                parsed = self.generate_structured(prompt, max_retries=1, max_tokens=attempt_tokens, **call_kwargs)
                print('\n=== BEFORE PYDANTIC VALIDATION ===')
                print(json.dumps(parsed, indent=2, default=str))
                parsed = _normalize_typed_payload(parsed, schema)
                validated = schema.model_validate(parsed)
                print('\n=== AFTER PYDANTIC VALIDATION ===')
                print(json.dumps(validated.model_dump(), indent=2, default=str))
                # Typed payload accumulation is owned by the guided generate_typed hook
                # (avoids double-append into the per-call store).
                schema_name = getattr(schema, '__name__', '') or ''
                if schema_name == 'EntitiesResponse':
                    print(f'[stage] pydantic_validation entities={len(getattr(validated, "entities", []) or [])}')
                elif schema_name == 'RelationsResponse':
                    print(f'[stage] pydantic_validation relations={len(getattr(validated, "relations", []) or [])}')
                elif schema_name == 'TripletsResponse':
                    print(f'[stage] pydantic_validation triplets={len(getattr(validated, "triplets", []) or [])}')
                return validated
            except (ProcessingError, ValidationError, Exception) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(min(8, 2 ** attempt))
                    continue
        raise ProcessingError(f'Failed to generate typed output after {retries} attempts: {last_error}')


# Ensure Semantica can resolve provider='bedrock' even before FactExtractor init.
provider_registry.register('bedrock', BedrockProvider)

# ---------------------------------------------------------------------------
# 5) Shared helpers — text, typed payloads, diagnostics
# ---------------------------------------------------------------------------
def collapse(text: str) -> str:
    return ' '.join((text or '').split())


def _normalize_typed_payload(parsed: Any, schema: Type[BaseModel]) -> dict:
    """Wrap bare lists/single items into Entities/Relations/TripletsResponse shape.

    Semantica's EntitiesResponse defaults missing `entities` to []. A single
    entity object (common when JSON is truncated) must not validate as empty.
    """
    name = getattr(schema, '__name__', '') or ''
    wrapper = {
        'EntitiesResponse': 'entities',
        'RelationsResponse': 'relations',
        'RelationsWithTemporalResponse': 'relations',
        'TripletsResponse': 'triplets',
    }.get(name)

    if wrapper is None:
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
        raise ProcessingError('Expected a JSON object for typed output')

    if isinstance(parsed, dict) and wrapper in parsed:
        return parsed

    if isinstance(parsed, list):
        return {wrapper: parsed}

    if isinstance(parsed, dict):
        # Single item returned without the wrapper key.
        item_markers = ('text', 'label', 'subject', 'predicate', 'object')
        if any(key in parsed for key in item_markers):
            return {wrapper: [parsed]}
        raise ProcessingError(f'Typed JSON missing "{wrapper}" list for {name}')

    raise ProcessingError(f'Unexpected typed JSON payload for {name}')


# LLM metadata Semantica drops when converting EntityOut → Entity.
_PRESERVE_META_KEYS = (
    'kind',
    'subtype',
    'identifiers',
    'aliases',
    'properties',
    'evidence',
)
# Per-call / context-local typed payloads (avoids concurrent mixing).
_TYPED_PAYLOADS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    'semantica_typed_payloads',
    default=None,
)


def _typed_store(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    if existing is not None:
        _TYPED_PAYLOADS.set(existing)
        return existing
    store = _TYPED_PAYLOADS.get()
    if store is None:
        store = {}
        _TYPED_PAYLOADS.set(store)
    return store


def _typed_set(schema_name: str, payload: Any, store: dict[str, Any] | None = None) -> None:
    """Accumulate typed payloads across chunks (never overwrite prior chunk results)."""
    target = _typed_store(store)
    existing = target.get(schema_name)
    if existing is None:
        target[schema_name] = payload
    elif isinstance(existing, list):
        existing.append(payload)
    else:
        target[schema_name] = [existing, payload]


def _typed_pop(schema_name: str, store: dict[str, Any] | None = None) -> Any:
    target = store if store is not None else (_TYPED_PAYLOADS.get() or {})
    return target.pop(schema_name, None)


def _flatten_typed_payloads(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        out: list[Any] = []
        for item in payload:
            out.extend(_flatten_typed_payloads(item))
        return out
    return [payload]


def _typed_row_count(payload: Any, key: str) -> int:
    return len(_typed_rows(payload, key)) if payload is not None else 0


# Pipeline stage diagnostics (per extract() call).
_EXTRACT_DIAGNOSTICS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    'extract_diagnostics',
    default=None,
)


def _diag_begin() -> list[dict[str, Any]]:
    bucket: list[dict[str, Any]] = []
    _EXTRACT_DIAGNOSTICS.set(bucket)
    return bucket


def _diag_record(event: str, **fields: Any) -> None:
    bucket = _EXTRACT_DIAGNOSTICS.get()
    if bucket is None:
        return
    row = {'event': event, **fields}
    bucket.append(row)
    # Compact console breadcrumb for operators.
    extra = ' '.join(f'{k}={v}' for k, v in fields.items() if k != 'dropped')
    print(f'[diag] {event}' + (f' {extra}' if extra else ''), flush=True)
    dropped = fields.get('dropped')
    if isinstance(dropped, list):
        for item in dropped[:20]:
            print(f'[diag]   drop: {item}', flush=True)


def _diag_snapshot() -> list[dict[str, Any]]:
    return list(_EXTRACT_DIAGNOSTICS.get() or [])

def _merge_preserved_metadata(base: dict | None, llm_meta: dict | None) -> dict:
    """Merge custom LLM fields into Semantica provenance without overwriting it."""
    out: dict[str, Any] = dict(base or {})
    src = llm_meta if isinstance(llm_meta, dict) else {}
    for key in _PRESERVE_META_KEYS:
        if key not in src:
            continue
        value = src[key]
        if value is None or value == '' or value == {} or value == []:
            continue
        if key == 'evidence' and isinstance(value, dict):
            existing = out.get('evidence') if isinstance(out.get('evidence'), dict) else {}
            merged_ev = dict(existing)
            for ek, ev in value.items():
                if ek not in merged_ev or merged_ev[ek] in (None, '', {}, []):
                    merged_ev[ek] = ev
            out['evidence'] = merged_ev
        elif key not in out or out[key] in (None, '', {}, []):
            out[key] = value
    return out

# ---------------------------------------------------------------------------
# 6) Document prep — section chunks + repeated banner collapse for NER
# ---------------------------------------------------------------------------
_HTML_TABLE = re.compile(r'<table\b[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
_PAGE_MARKER = re.compile(r'<!--\s*page\s+\d+\s*-->', re.IGNORECASE)
_HTML_TAG = re.compile(r'<[^>]+>')
_MARKDOWN_HEADING = re.compile(r'(?m)^(#{1,6}\s+\S.*)$')
_DOC_NER_CONTEXT = threading.local()

def _boilerplate_detection_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load table_extraction.boilerplate_detection from identity_resolution.yaml."""
    pol = policy if policy is not None else load_policy()
    table = (pol.data.get('table_extraction') or {}) if isinstance(pol.data, dict) else {}
    cfg = table.get('boilerplate_detection') if isinstance(table, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    hints = [
        collapse(str(h or '')).casefold()
        for h in (cfg.get('content_table_hints') or [])
        if collapse(str(h or ''))
    ]
    return {
        'content_table_hints': hints,
        'min_content_rows': int(cfg.get('min_content_rows') or 6),
        'min_content_plain_chars': int(cfg.get('min_content_plain_chars') or 500),
        'max_banner_rows': int(cfg.get('max_banner_rows') or 4),
        'max_banner_plain_chars': int(cfg.get('max_banner_plain_chars') or 400),
        'min_repetitions': int(cfg.get('min_repetitions') or 2),
    }


def _table_plain_text(html: str) -> str:
    return collapse(_HTML_TAG.sub(' ', html or ''))


def _boilerplate_fingerprint(block: str) -> str:
    """Structure fingerprint: digits neutralized so page/sheet indices can repeat."""
    plain = _table_plain_text(block)
    plain = re.sub(r'\d+', '#', plain)
    return plain.casefold()


def _table_row_count(html: str) -> int:
    return len(re.findall(r'<tr\b', html or '', flags=re.IGNORECASE))


def _looks_like_content_table(html: str, *, settings: dict[str, Any] | None = None) -> bool:
    """True when the table likely carries tender/spec/commercial payload (never boilerplate)."""
    cfg = settings if settings is not None else _boilerplate_detection_settings()
    plain = _table_plain_text(html).casefold()
    if not plain:
        return False
    hints = cfg.get('content_table_hints') or []
    if any(hint in plain for hint in hints):
        return True
    if _table_row_count(html) >= int(cfg.get('min_content_rows') or 6):
        return True
    if len(plain) >= int(cfg.get('min_content_plain_chars') or 500):
        return True
    return False


def _banner_role_for_span(document_text: str, start: int, end: int) -> str:
    prefix = document_text[max(0, start - 80) : start]
    if _PAGE_MARKER.search(prefix):
        return 'page_header'
    suffix = document_text[end : min(len(document_text), end + 80)]
    if _PAGE_MARKER.search(suffix):
        return 'page_footer'
    return 'document_metadata_banner'


def detect_repeated_boilerplate_spans(
    document_text: str,
    *,
    policy: Policy | None = None,
) -> list[dict[str, Any]]:
    """Identify repeated page headers/footers/metadata banners via structure + repetition.

    Returns exact source spans with positions. Unique or uncertain blocks are preserved.
    Content protection uses policy table_extraction.boilerplate_detection + prompt A6b.
    """
    text = document_text or ''
    if not text:
        return []
    cfg = _boilerplate_detection_settings(policy)

    candidates: list[dict[str, Any]] = []
    for match in _HTML_TABLE.finditer(text):
        html = match.group(0)
        start, end = match.start(), match.end()
        fp = _boilerplate_fingerprint(html)
        if not fp or len(fp) < 12:
            continue
        candidates.append({
            'source_text': html,
            'start_char': start,
            'end_char': end,
            'fingerprint': fp,
            'row_count': _table_row_count(html),
            'plain_len': len(_table_plain_text(html)),
            'content_like': _looks_like_content_table(html, settings=cfg),
            'role': _banner_role_for_span(text, start, end),
        })

    by_fp: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_fp.setdefault(row['fingerprint'], []).append(row)

    min_reps = int(cfg.get('min_repetitions') or 2)
    max_rows = int(cfg.get('max_banner_rows') or 4)
    max_plain = int(cfg.get('max_banner_plain_chars') or 400)

    spans: list[dict[str, Any]] = []
    for fp, group in by_fp.items():
        group = sorted(group, key=lambda r: int(r['start_char']))
        rep = len(group)
        for index, row in enumerate(group):
            content_like = bool(row['content_like'])
            short_meta = int(row['row_count']) <= max_rows and int(row['plain_len']) <= max_plain
            high_conf = rep >= min_reps and short_meta and not content_like
            if content_like or rep < min_reps:
                action = 'preserve'
                confidence = 'uncertain' if rep >= min_reps and content_like else 'high'
                keep_reason = 'content_or_tender_table' if content_like else 'unique_or_unrepeated'
            elif high_conf and index == 0:
                action = 'preserve'
                confidence = 'high'
                keep_reason = 'first_exemplar_kept'
            elif high_conf:
                action = 'omit_repeat'
                confidence = 'high'
                keep_reason = None
            else:
                action = 'preserve'
                confidence = 'uncertain'
                keep_reason = 'uncertain_structure'

            spans.append({
                'source_text': row['source_text'],
                'start_char': row['start_char'],
                'end_char': row['end_char'],
                'role': row['role'],
                'fingerprint': fp,
                'repetition_count': rep,
                'confidence': confidence,
                'action': action,
                'keep_reason': keep_reason,
            })

    spans.sort(key=lambda s: int(s['start_char']))
    return spans


def verify_boilerplate_candidates(
    document_text: str,
    candidates: list[dict[str, Any]] | None,
    *,
    policy: Policy | None = None,
) -> list[dict[str, Any]]:
    """Python verification of prompt-proposed boilerplate spans (exact source + repetition)."""
    text = document_text or ''
    detected = detect_repeated_boilerplate_spans(text, policy=policy)
    by_pos = {(int(s['start_char']), int(s['end_char'])): s for s in detected}
    verified: list[dict[str, Any]] = []
    for raw in candidates or []:
        if not isinstance(raw, dict):
            continue
        quote = raw.get('source_text')
        if not isinstance(quote, str) or not quote or quote not in text:
            continue
        start = raw.get('start_char')
        end = raw.get('end_char')
        if not (isinstance(start, int) and isinstance(end, int) and text[start:end] == quote):
            spans_found = _find_all_spans(text, quote)
            if len(spans_found) != 1:
                continue
            start, end = spans_found[0]
        hit = by_pos.get((start, end))
        if hit is None:
            hit = next((s for s in detected if s['source_text'] == quote), None)
        if hit is None or hit.get('confidence') != 'high':
            continue
        out = dict(hit)
        if raw.get('role'):
            out['role'] = collapse(str(raw.get('role')))
        out['verified_from_prompt'] = True
        verified.append(out)
    return verified


def _doc_boilerplate_spans(*, policy: Policy | None = None) -> list[dict[str, Any]] | None:
    """Full-document boilerplate spans stored for the active extract() call."""
    spans = getattr(_DOC_NER_CONTEXT, 'spans', None)
    if isinstance(spans, list):
        return spans
    return None


def _chunk_absolute_start(full_document: str, chunk: str) -> int | None:
    if not full_document or not chunk:
        return None
    explicit = getattr(_DOC_NER_CONTEXT, 'chunk_start', None)
    if isinstance(explicit, int) and explicit >= 0:
        end = explicit + len(chunk)
        if full_document[explicit:end] == chunk:
            return explicit
    idx = full_document.find(chunk)
    if idx < 0:
        return None
    if full_document.find(chunk, idx + 1) >= 0:
        return None
    return idx


def _omit_boilerplate_in_text(
    text: str,
    spans: list[dict[str, Any]],
    *,
    absolute_start: int = 0,
) -> str:
    """Omit high-confidence repeated banners; keep first exemplar and all content tables."""
    if not text or not spans:
        return text

    exemplar_starts = {
        (str(s.get('fingerprint') or ''), int(s['start_char']))
        for s in spans
        if s.get('keep_reason') == 'first_exemplar_kept' and s.get('fingerprint')
    }
    omit_fps = {
        str(s.get('fingerprint') or '')
        for s in spans
        if s.get('action') == 'omit_repeat' and s.get('confidence') == 'high' and s.get('fingerprint')
    }
    if not omit_fps:
        return text

    local_omit: list[tuple[int, int, dict[str, Any]]] = []
    # Prefer exact absolute omit_repeat spans when the chunk window is known.
    for span in spans:
        if span.get('action') != 'omit_repeat' or span.get('confidence') != 'high':
            continue
        local_start = int(span['start_char']) - absolute_start
        local_end = int(span['end_char']) - absolute_start
        if local_start < 0 or local_end > len(text) or local_start >= local_end:
            continue
        if text[local_start:local_end] != span.get('source_text'):
            continue
        local_omit.append((local_start, local_end, span))

    if not local_omit:
        for match in _HTML_TABLE.finditer(text):
            html = match.group(0)
            fp = _boilerplate_fingerprint(html)
            if fp not in omit_fps:
                continue
            abs_pos = absolute_start + match.start()
            if (fp, abs_pos) in exemplar_starts:
                continue
            local_omit.append((
                match.start(),
                match.end(),
                {'role': 'page_header', 'fingerprint': fp},
            ))

    if not local_omit:
        return text
    local_omit.sort(key=lambda row: row[0])
    parts: list[str] = []
    last = 0
    for start, end, span in local_omit:
        if start < last:
            continue
        parts.append(text[last:start])
        parts.append(
            f'\n<!-- repeated page banner omitted '
            f'start_char={absolute_start + start} end_char={absolute_start + end} '
            f'role={span.get("role")} -->\n'
        )
        last = end
    parts.append(text[last:])
    return ''.join(parts)


def _ner_context_text(chunk: str, *, policy: Policy | None = None) -> str:
    """Collapse repeated boilerplate banners in NER prompt SOURCE only (offsets stay on full doc)."""
    text = chunk or ''
    if not text:
        return text

    full_document = getattr(_DOC_NER_CONTEXT, 'text', None)
    doc_spans = _doc_boilerplate_spans(policy=policy)
    if isinstance(full_document, str) and doc_spans is not None:
        abs_start = _chunk_absolute_start(full_document, text)
        if abs_start is None:
            abs_start = int(getattr(_DOC_NER_CONTEXT, 'chunk_start', 0) or 0)
        focused = _omit_boilerplate_in_text(text, doc_spans, absolute_start=abs_start)
    else:
        spans = detect_repeated_boilerplate_spans(text, policy=policy)
        focused = _omit_boilerplate_in_text(text, spans, absolute_start=0)

    heading = collapse(str(getattr(_DOC_NER_CONTEXT, 'parent_heading', '') or ''))
    if heading:
        focused = f'<!-- section context: {heading} -->\n{focused}'
    return focused


def _table_ranges(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _HTML_TABLE.finditer(text or '')]


def _parent_heading_before(text: str, pos: int) -> str:
    source = text or ''
    at_pos = _MARKDOWN_HEADING.match(source[max(0, pos):])
    if at_pos:
        return collapse(at_pos.group(1))
    headings = list(_MARKDOWN_HEADING.finditer(source[: max(0, pos)]))
    if not headings:
        return ''
    return collapse(headings[-1].group(1))


def _markdown_recursive_separators() -> list[str]:
    """Separator priority for Semantica recursive splitting on engineering Markdown."""
    return [
        '\n## ',
        '\n### ',
        '\n#### ',
        '\n<!-- page ',
        '\n\n',
        '\n',
        ' ',
        '',
    ]


def _repair_table_boundaries(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand/merge spans so HTML tables are never cut across chunk boundaries."""
    source = text or ''
    tables = _table_ranges(source)
    if not spans:
        return []
    if not tables:
        return list(spans)

    repaired: list[dict[str, Any]] = []
    for span in spans:
        start = int(span['start'])
        end = int(span['end'])
        for t_start, t_end in tables:
            if end <= t_start or start >= t_end:
                continue
            if t_start < start < t_end:
                start = t_start
            if t_start < end < t_end:
                end = t_end
        start = max(0, min(start, len(source)))
        end = max(start, min(end, len(source)))
        repaired.append({
            'text': source[start:end],
            'start': start,
            'end': end,
            'parent_heading': span.get('parent_heading') or _parent_heading_before(source, start),
            'method': span.get('method') or 'recursive',
        })

    repaired.sort(key=lambda row: int(row['start']))
    merged: list[dict[str, Any]] = []
    for span in repaired:
        if merged and int(span['start']) < int(merged[-1]['end']):
            prev = merged[-1]
            new_end = max(int(prev['end']), int(span['end']))
            merged[-1] = {
                'text': source[int(prev['start']):new_end],
                'start': int(prev['start']),
                'end': new_end,
                'parent_heading': prev.get('parent_heading') or span.get('parent_heading') or '',
                'method': prev.get('method') or span.get('method') or 'recursive',
            }
        else:
            merged.append(span)
    return merged


def section_aware_chunk_spans(text: str, max_chars: int) -> list[dict[str, Any]]:
    """Chunk large docs via Semantica recursive split + engineering safeguards.

    Uses Semantica TextSplitter(method='recursive') with Markdown/page separators.
    Hierarchical/structural Semantica methods are not used alone here: hierarchical
    drops heading text/offsets, and structural rejoins text so evidence offsets break.
    Safeguards: exact document slices, intact HTML tables, parent heading context.
    """
    source = text or ''
    limit = int(max_chars or 12000)
    if limit < 1:
        limit = 12000
    if not source:
        return []
    if len(source) <= limit:
        return [{
            'text': source,
            'start': 0,
            'end': len(source),
            'parent_heading': _parent_heading_before(source, 0),
            'method': 'passthrough',
        }]

    available = set(list_split_methods() or [])
    if 'recursive' not in available:
        raise ProcessingError(
            f"Semantica recursive splitter unavailable; have: {sorted(available)}"
        )

    # Call split_recursive directly. TextSplitter treats chunk_overlap=0 as falsy
    # and silently substitutes config default 200, which creates tiny crawl-steps.
    chunks = split_recursive(
        source,
        chunk_size=limit,
        chunk_overlap=0,
        separators=_markdown_recursive_separators(),
    )

    spans: list[dict[str, Any]] = []
    for chunk in chunks or []:
        start = int(getattr(chunk, 'start_index', 0) or 0)
        end = int(getattr(chunk, 'end_index', start) or start)
        start = max(0, min(start, len(source)))
        end = max(start, min(end, len(source)))
        # Prefer exact source slice over Semantica's stripped chunk.text for offsets.
        slice_text = source[start:end]
        if not slice_text.strip():
            continue
        spans.append({
            'text': slice_text,
            'start': start,
            'end': end,
            'parent_heading': _parent_heading_before(source, start),
            'method': 'recursive',
        })

    spans = _repair_table_boundaries(source, spans)
    if not spans:
        return [{
            'text': source,
            'start': 0,
            'end': len(source),
            'parent_heading': _parent_heading_before(source, 0),
            'method': 'passthrough',
        }]
    return spans


def _shift_extract_offsets(rows: list, delta: int) -> list:
    """Shift Semantica start/end offsets from chunk-local to document-absolute."""
    if not delta:
        return list(rows or [])
    shifted: list = []
    for row in rows or []:
        start = getattr(row, 'start_char', None)
        end = getattr(row, 'end_char', None)
        if isinstance(start, int):
            try:
                row.start_char = start + delta
            except Exception:
                pass
        if isinstance(end, int):
            try:
                row.end_char = end + delta
            except Exception:
                pass
        meta = getattr(row, 'metadata', None)
        if isinstance(meta, dict):
            evidence = meta.get('evidence')
            if isinstance(evidence, dict):
                for key in ('start_char', 'end_char'):
                    value = evidence.get(key)
                    if isinstance(value, int):
                        evidence[key] = value + delta
        shifted.append(row)
    return shifted


def _row_metadata(row: Any) -> dict:
    """Return a mutable metadata dict for a Semantica object or plain dict row."""
    if isinstance(row, dict):
        meta = row.get('metadata')
        if not isinstance(meta, dict):
            meta = {}
            row['metadata'] = meta
        return meta
    meta = getattr(row, 'metadata', None)
    if not isinstance(meta, dict):
        meta = {}
        try:
            row.metadata = meta
        except Exception:
            return {}
    return meta


def _stamp_chunk_provenance(
    rows: list,
    *,
    source_file: str | None = None,
    chunk_start: int = 0,
    chunk_end: int | None = None,
    parent_heading: str = '',
    chunk_method: str = '',
) -> list:
    """Attach chunk/section provenance at emit time (survives merge/serialize)."""
    for row in rows or []:
        meta = _row_metadata(row)
        if not isinstance(meta, dict):
            continue
        provenance = meta.get('provenance') if isinstance(meta.get('provenance'), dict) else {}
        provenance = dict(provenance)
        if source_file:
            provenance['source_file'] = source_file
        provenance['chunk_start'] = int(chunk_start or 0)
        if chunk_end is not None:
            provenance['chunk_end'] = int(chunk_end)
        heading = collapse(parent_heading or '')
        if heading:
            provenance['parent_heading'] = heading
        method = collapse(chunk_method or '')
        if method:
            provenance['chunk_method'] = method
        meta['provenance'] = provenance
    return list(rows or [])


def _finalize_chunk_evidence(
    rows: list,
    *,
    full_text: str,
    source_file: str | None = None,
    policy: Policy | None = None,
    kind: str = 'entity',
) -> list:
    """Verify LLM quotes against the full document; never invent evidence.

    Fills absolute offsets when unique. Optionally recovers a source-backed
    relation/triplet quote when missing. Does not auto-confirm graph_eligible.
    """
    pol = policy if policy is not None else load_policy()
    settings = _claim_evidence_recovery_settings(pol)
    require_creation = bool(settings.get('require_creation_evidence', True))
    doc = full_text or ''
    kind_key = collapse(kind or 'entity').casefold()

    for row in rows or []:
        meta = _row_metadata(row)
        if not isinstance(meta, dict):
            continue
        evidence = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else None
        quote = str((evidence or {}).get('source_text') or '') if evidence else ''
        subject = ''
        obj = ''
        predicate = ''
        if kind_key in {'relation', 'relations', 'triplet', 'triplets'}:
            if isinstance(row, dict):
                subject = endpoint_text(row.get('subject'))
                obj = endpoint_text(row.get('object'))
                predicate = collapse(str(row.get('predicate') or ''))
            else:
                subject = endpoint_text(getattr(row, 'subject', None))
                obj = endpoint_text(getattr(row, 'object', None))
                predicate = collapse(str(getattr(row, 'predicate', '') or ''))

        verified = _verify_evidence(evidence, doc, source_file=source_file) if evidence else None
        if verified:
            meta['evidence'] = verified
        else:
            if quote and quote not in doc:
                # Mismatched quote — drop fabricated evidence before recovery.
                meta.pop('evidence', None)
            need_recovery = bool(
                require_creation
                and kind_key in {'relation', 'relations', 'triplet', 'triplets'}
                and subject
                and obj
                and not isinstance(meta.get('evidence'), dict)
            )
            if need_recovery:
                recovered = _recover_claim_evidence_quote(
                    doc,
                    subject=subject,
                    obj=obj,
                    predicate=predicate,
                    attribute_mode=_looks_like_technical_attribute_claim(
                        predicate, obj, policy=pol,
                    ) if predicate and obj else False,
                    policy=pol,
                )
                if recovered:
                    verified = _verify_evidence(
                        {'source_text': recovered, 'source_file': source_file},
                        doc,
                        source_file=source_file,
                    )
                    if verified:
                        meta['evidence'] = verified
                        meta['evidence_recovery'] = 'deterministic_span'
            # Never invent; leave evidence absent when recovery fails.

        # Attribute-shaped triplets: value quote + unique section owner separately.
        if kind_key in {'triplet', 'triplets'} and predicate and obj:
            attr_mode = _looks_like_technical_attribute_claim(
                predicate, obj, evidence_quote=str((meta.get('evidence') or {}).get('source_text') or ''),
                document_text=doc, policy=pol,
            )
            if attr_mode and isinstance(meta.get('evidence'), dict):
                value_quote = str(meta['evidence'].get('source_text') or '')
                if value_quote and not meta.get('section_context'):
                    ownership = _classify_attribute_ownership(
                        doc,
                        meta['evidence'],
                        obj=obj,
                        policy=pol,
                    )
                    if isinstance(ownership, dict) and ownership.get('scope') == 'specific':
                        table_row = ownership.get('table_row')
                        if isinstance(table_row, TableOwnedRow) and table_row.owner_surface:
                            meta['section_context'] = _table_owner_provenance(
                                table_row,
                                source_file=source_file,
                                resolver=(
                                    'table_row_group'
                                    if table_row.is_continuation
                                    else 'table_row_owner'
                                ),
                            )
                            meta['attribute_source_scope'] = 'specific'
                        else:
                            section = ownership.get('section')
                            if isinstance(section, dict):
                                meta['section_context'] = _section_context_provenance(
                                    section,
                                    source_file=source_file,
                                    resolver='deterministic_section_row',
                                )
                                meta['attribute_source_scope'] = 'specific'
    return list(rows or [])


def _postprocess_chunk_extract(
    rows: list,
    *,
    full_text: str,
    source_file: str | None = None,
    span: dict[str, Any] | None = None,
    policy: Policy | None = None,
    kind: str = 'entity',
) -> list:
    """Shift surface offsets, stamp provenance, re-verify quotes on the full document.

    Evidence offsets are cleared before full-doc verify so chunk-local or
    pre-shifted values cannot double-apply; `_verify_evidence` fills absolute
    positions when the quote uniquely matches.
    """
    span = span or {}
    chunk_start = int(span.get('start') or 0)
    chunk_end = span.get('end')
    try:
        chunk_end_i = int(chunk_end) if chunk_end is not None else None
    except (TypeError, ValueError):
        chunk_end_i = None
    out = _shift_extract_offsets(list(rows or []), chunk_start)
    for row in out:
        meta = _row_metadata(row)
        if not isinstance(meta, dict):
            continue
        evidence = meta.get('evidence')
        if isinstance(evidence, dict):
            evidence['start_char'] = None
            evidence['end_char'] = None
    _stamp_chunk_provenance(
        out,
        source_file=source_file,
        chunk_start=chunk_start,
        chunk_end=chunk_end_i,
        parent_heading=str(span.get('parent_heading') or ''),
        chunk_method=str(span.get('method') or ''),
    )
    return _finalize_chunk_evidence(
        out,
        full_text=full_text,
        source_file=source_file,
        policy=policy,
        kind=kind,
    )

# ---------------------------------------------------------------------------
# 7) Evidence — verify quotes and attach claim evidence
# ---------------------------------------------------------------------------
def _find_all_spans(document_text: str, quote: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not document_text or not quote:
        return spans
    start = 0
    while True:
        idx = document_text.find(quote, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(quote)))
        start = idx + 1
    return spans


def _verify_evidence(
    evidence: Any,
    document_text: str,
    *,
    source_file: str | None = None,
) -> dict | None:
    """Keep verbatim evidence; set offsets only when uniquely/supplied-valid."""
    if not isinstance(evidence, dict):
        return None
    quote = evidence.get('source_text')
    if not isinstance(quote, str) or not quote:
        return None
    doc = document_text or ''
    if quote not in doc:
        return None
    file_name = source_file or evidence.get('source_file') or None
    if isinstance(file_name, str):
        file_name = file_name.strip() or None

    supplied_start = evidence.get('start_char')
    supplied_end = evidence.get('end_char')
    if isinstance(supplied_start, int) and isinstance(supplied_end, int):
        if 0 <= supplied_start < supplied_end <= len(doc) and doc[supplied_start:supplied_end] == quote:
            return {
                'source_text': quote,
                'source_file': file_name,
                'start_char': supplied_start,
                'end_char': supplied_end,
            }

    spans = _find_all_spans(doc, quote)
    if len(spans) == 1:
        start, end = spans[0]
        return {
            'source_text': quote,
            'source_file': file_name,
            'start_char': start,
            'end_char': end,
        }

    # Ambiguous or unresolvable occurrence: keep quote, null offsets.
    return {
        'source_text': quote,
        'source_file': file_name,
        'start_char': None,
        'end_char': None,
    }

_SECTION_AWARE_CACHE: dict[str, dict[str, Any]] = {}
_DOC_KV_LABEL_CACHE: dict[tuple[str, str], list[str]] = {}
_DOC_SECTION_INDEX_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}
_DOC_TABLE_OWNER_INDEX_CACHE: dict[tuple[str, str], 'TableOwnershipIndex'] = {}
# Optional injectable owner picker: (payload: dict) -> section_number|tag|None
_OWNER_RESOLVER_HOOK: Callable[..., str | None] | None = None


def clear_section_aware_caches() -> None:
    """Drop policy/document caches (tests + policy reload)."""
    _SECTION_AWARE_CACHE.clear()
    _DOC_KV_LABEL_CACHE.clear()
    _DOC_SECTION_INDEX_CACHE.clear()
    _DOC_TABLE_OWNER_INDEX_CACHE.clear()


def _policy_cache_token(policy: Policy | None = None) -> str:
    pol = policy if policy is not None else load_policy()
    path = getattr(pol, 'path', None) or POLICY_PATH
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        mtime = 0
    return f'{path}:{mtime}'


def _document_cache_token(document_text: str) -> str:
    # Stable within-process digest; avoids Python's salted hash() across runs in tests.
    return hashlib.sha1((document_text or '').encode('utf-8', errors='surrogatepass')).hexdigest()


def _tokenize_surface(text: str) -> list[str]:
    raw = collapse(text or '')
    if not raw:
        return []
    for ch in '(),[];:|/\\':
        raw = raw.replace(ch, ' ')
    return [tok for tok in raw.split() if tok]


def _endpoint_supported_in_quote(
    endpoint: str,
    folded_quote: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """True when full surface, engineering tag, or a meaningful identity token appears."""
    surface = collapse(endpoint or '')
    if not surface or not folded_quote:
        return False
    if surface.casefold() in folded_quote:
        return True
    for tag in _engineering_tags_in(surface, policy=policy):
        norm = _normalize_equipment_tag(tag)
        if norm and norm in folded_quote:
            return True
    # Prefer longer identity-bearing tokens; avoid short glue words.
    tokens = [tok.casefold() for tok in _tokenize_surface(surface) if len(tok) >= 6]
    return any(tok in folded_quote for tok in tokens)


def _token_parts(text: str) -> set[str]:
    return {tok for tok in re.split(r'[^a-z0-9]+', collapse(text or '').casefold()) if tok}


# Short linguistic glue ignored when matching predicates to field-label tokens.
_FIELD_TOKEN_STOPWORDS = frozenset({'of', 'to', 'in', 'at'})


def _compile_regex(pattern: str, flags: int = 0) -> re.Pattern[str] | None:
    raw = str(pattern or '').strip()
    if not raw:
        return None
    try:
        return re.compile(raw, flags)
    except re.error:
        return None


def _extract_named_group_body(pattern: str, name: str = 'tag') -> str:
    token = f'(?P<{name}>'
    idx = (pattern or '').find(token)
    if idx < 0:
        return ''
    start = idx + len(token)
    depth = 1
    i = start
    while i < len(pattern) and depth:
        ch = pattern[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return pattern[start:i]
        i += 1
    return ''


def _engineering_tag_core_from_pattern(pattern: str) -> str:
    """Unwrap an anchored identifier pattern into a reusable tag-body regex."""
    tag_core = str(pattern or '').strip()
    if not tag_core:
        return ''
    named = _extract_named_group_body(tag_core, 'tag')
    if named:
        return named
    tag_core = re.sub(r'^\(\?i\)', '', tag_core)
    tag_core = tag_core.replace(r'\b', '')
    tag_core = tag_core.strip('^$')
    return tag_core


def _parse_heading_body(body: str) -> dict[str, str]:
    """Split a generic heading body into optional section number, head, and title."""
    text = collapse(body or '')
    num = ''
    rest = text
    parts = text.split(None, 1)
    if parts and all(ch.isdigit() or ch == '.' for ch in parts[0]) and '.' in parts[0]:
        num = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
    head = rest
    title = ''
    if ':' in rest:
        left, right = rest.split(':', 1)
        head = collapse(left)
        title = collapse(right)
    return {'section_number': num, 'head': head, 'title': title}


def _match_trailing_unit_suffix(text: str) -> str:
    """Return trailing '(unit)' / '[unit]' suffix when present (generic syntax)."""
    raw = text or ''
    if not raw:
        return ''
    i = 0
    while i < len(raw) and raw[i].isspace():
        i += 1
    if i >= len(raw) or raw[i] not in '([':
        return ''
    open_ch = raw[i]
    close_ch = ')' if open_ch == '(' else ']'
    depth = 0
    j = i
    while j < len(raw):
        if raw[j] == open_ch:
            depth += 1
        elif raw[j] == close_ch:
            depth -= 1
            if depth == 0:
                return raw[: j + 1]
        j += 1
    return ''


def _split_pipe_row(line: str) -> list[str] | None:
    """Generic Markdown pipe-table row -> cells (None if not a pipe row)."""
    stripped = (line or '').strip()
    if '|' not in stripped:
        return None
    if stripped.startswith('#'):
        return None
    # Require a pipe-led or pipe-containing table row with at least two cells.
    raw = stripped.strip('|')
    cells = [collapse(c) for c in raw.split('|')]
    if len(cells) < 2:
        return None
    if all(set(c) <= {'-', ':', ' '} for c in cells):
        return None  # separator row
    return cells


def _iter_atx_markdown_headings(document_text: str) -> list[dict[str, Any]]:
    """Generic ATX Markdown headings (# … ######)."""
    text = document_text or ''
    out: list[dict[str, Any]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        raw = line.rstrip('\r\n')
        stripped = raw.lstrip()
        if not stripped.startswith('#'):
            continue
        level = 0
        while level < len(stripped) and stripped[level] == '#':
            level += 1
        if level < 1 or level > 6:
            continue
        if level < len(stripped) and not stripped[level].isspace():
            continue
        body = stripped[level:].strip()
        if not body:
            continue
        parsed = _parse_heading_body(body)
        out.append({
            **parsed,
            'heading_start': line_start + raw.find('#'),
            'heading_end': line_start + len(raw),
            'kind': 'markdown',
        })
    return out


def _iter_html_headings(document_text: str) -> list[dict[str, Any]]:
    """Generic HTML h1–h6 headings via tag scan (not document-specific)."""
    text = document_text or ''
    out: list[dict[str, Any]] = []
    lower = text.casefold()
    pos = 0
    while True:
        start = -1
        tag_name = ''
        for level in range(1, 7):
            token = f'<h{level}'
            idx = lower.find(token, pos)
            if idx < 0:
                continue
            if start < 0 or idx < start:
                start = idx
                tag_name = f'h{level}'
        if start < 0:
            break
        gt = text.find('>', start)
        if gt < 0:
            break
        close = lower.find(f'</{tag_name}>', gt + 1)
        if close < 0:
            pos = gt + 1
            continue
        end = close + len(f'</{tag_name}>')
        inner = text[gt + 1:close]
        plain = collapse(_HTML_TAG.sub(' ', inner))
        parsed = _parse_heading_body(plain)
        out.append({
            **parsed,
            'heading_start': start,
            'heading_end': end,
            'kind': 'html',
        })
        pos = end
    return out


def _iter_bold_headings(document_text: str) -> list[dict[str, Any]]:
    """Single-line bold Markdown headings: **body**."""
    text = document_text or ''
    out: list[dict[str, Any]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        raw = line.rstrip('\r\n').strip()
        if len(raw) < 4 or not raw.startswith('**') or not raw.endswith('**'):
            continue
        if raw.count('**') != 2:
            continue
        body = raw[2:-2].strip()
        if not body or body.startswith('#'):
            continue
        parsed = _parse_heading_body(body)
        abs_start = line_start + line.find('**')
        out.append({
            **parsed,
            'heading_start': abs_start,
            'heading_end': abs_start + len(raw),
            'kind': 'bold',
        })
    return out


def _leading_section_number(text: str, end: int) -> tuple[str, int] | None:
    """If text[:end] ends with outline number 'N.N', return (number, start_index).

    Rejects numeric glue from prior values (e.g. '...1104.2 TAG') while allowing
    unit/letter mash ('...°C6.2 TAG') and punctuation ('...hopper.6.2 TAG').
    """
    i = end
    while i > 0 and text[i - 1].isspace():
        i -= 1
    window_start = max(0, i - 16)
    window = text[window_start:i]
    matches = list(re.finditer(r'(\d{1,3}(?:\.\d{1,3})+)$', window))
    if not matches:
        return None
    match = matches[-1]
    token = match.group(1)
    start = window_start + match.start(1)
    if start > 0 and text[start - 1].isdigit():
        return None
    if '.' not in token:
        return None
    return token, start


def _iter_inline_numbered_equipment_headings(
    document_text: str,
    *,
    policy: Policy | None = None,
    boundary_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find mashed/HTML 'N.N TAG: title' starts using ontology tag patterns only."""
    text = document_text or ''
    settings = _section_aware_evidence_settings(policy)
    tag_re = settings.get('engineering_tag_re')
    if tag_re is None or not text:
        return []
    seps = tuple(settings.get('kv_separators') or _kv_separators(policy))
    labels = list(boundary_labels or [])
    out: list[dict[str, Any]] = []
    for match in tag_re.finditer(text):
        tag_start = match.start()
        # Head may be "TAG" or "TAG and TAG".
        head_start = tag_start
        head_end = match.end()
        # Extend through " and <tag>"
        cursor = head_end
        while True:
            rest = text[cursor:]
            m_and = re.match(r'(\s+and\s+)', rest, flags=re.IGNORECASE)
            if not m_and:
                break
            nxt = tag_re.match(text, cursor + m_and.end())
            if not nxt:
                break
            cursor = nxt.end()
            head_end = cursor
        # Require ":" after head.
        after = text[head_end:]
        colon = re.match(r'\s*:\s*', after)
        if not colon:
            continue
        title_start = head_end + colon.end()
        # Require numbered section immediately before the tag head.
        numbered = _leading_section_number(text, head_start)
        if numbered is None:
            continue
        num, num_start = numbered
        # Skip ATX/bold heading lines (handled by structural parsers).
        line_start = text.rfind('\n', 0, num_start) + 1
        prefix = text[line_start:num_start].lstrip()
        if prefix.startswith('#') or prefix.startswith('**'):
            continue
        title_region = text[title_start: title_start + 240]
        title_end = title_start + len(title_region)
        kv_hits = _find_label_separator_hits(
            title_region, labels, seps, policy=policy,
        )
        if kv_hits:
            title_end = title_start + kv_hits[0][0]
        else:
            # Stop at next numbered tag heading if present in the window.
            for nxt in tag_re.finditer(text, title_start):
                prev = _leading_section_number(text, nxt.start())
                if prev is not None and prev[0] != num:
                    title_end = min(title_end, prev[1])
                    break
        title = text[title_start:title_end]
        head = text[head_start:head_end]
        out.append({
            'section_number': num,
            'head': collapse(head),
            'title': collapse(title),
            'heading_start': num_start,
            'heading_end': title_end,
            'kind': 'inline',
        })
    # Deduplicate by heading_start.
    deduped: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in sorted(out, key=lambda r: int(r['heading_start'])):
        start = int(row['heading_start'])
        if start in seen:
            continue
        seen.add(start)
        deduped.append(row)
    return deduped


def _kv_separators(policy: Policy | None = None) -> tuple[str, ...]:
    """Load separators from table_extraction.inline_kv_separators (YAML only)."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    table = data.get('table_extraction') if isinstance(data.get('table_extraction'), dict) else {}
    raw = table.get('inline_kv_separators') if isinstance(table, dict) else None
    seps: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        sep = str(item or '')
        if not sep or sep in seen:
            continue
        seen.add(sep)
        seps.append(sep)
    # Longest first so ' - ' wins over shorter dashes and ': ' over ':'.
    seps.sort(key=len, reverse=True)
    return tuple(seps)


def _is_plausible_kv_label(label: str) -> bool:
    text = collapse(label or '')
    if not text or len(text) > 80:
        return False
    if _PAGE_MARKER.search(text) or '<' in text or '>' in text:
        return False
    if not re.match(r'^[A-Za-z]', text):
        return False
    if not re.search(r'[A-Za-z]', text):
        return False
    # Reject sentence-like blobs.
    if text.count(' ') > 10:
        return False
    return True


class _HtmlTableCellCollector(HTMLParser):
    """Collect table rows as plain cell texts via stdlib HTML parsing (no tag regex).

    Tracks rowspan/colspan so logical grids can be reconstructed. Offset-aware
    ownership uses a separate document scan; this collector remains text-first.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row_metas: list[list[dict[str, Any]]] = []
        self._row: list[str] | None = None
        self._row_meta: list[dict[str, Any]] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_depth = 0
        self._pending_span: tuple[int, int] = (1, 1)
        self._pending_is_header = False
        # Active rowspans from prior rows: col_index -> (remaining_rows, text, is_header)
        self._rowspan_carry: dict[int, tuple[int, str, bool]] = {}

    @staticmethod
    def _span_int(attrs: list[tuple[str, str | None]], name: str) -> int:
        for key, value in attrs or []:
            if collapse(str(key or '')).casefold() != name:
                continue
            try:
                return max(1, int(str(value or '1')))
            except (TypeError, ValueError):
                return 1
        return 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = (tag or '').casefold()
        if name == 'tr':
            self._row = []
            self._row_meta = []
        elif name in {'td', 'th'} and self._row is not None:
            if self._cell_parts is None:
                self._cell_parts = []
                self._cell_depth = 1
                self._pending_span = (
                    self._span_int(attrs, 'rowspan'),
                    self._span_int(attrs, 'colspan'),
                )
                self._pending_is_header = name == 'th'
            else:
                self._cell_depth += 1

    def _emit_carried_cells(self) -> None:
        if self._row is None or self._row_meta is None:
            return
        col = len(self._row)
        while col in self._rowspan_carry:
            remaining, text, is_header = self._rowspan_carry[col]
            self._row.append(text)
            self._row_meta.append({
                'text': text,
                'rowspan': 1,
                'colspan': 1,
                'is_header': is_header,
                'from_rowspan': True,
            })
            if remaining <= 1:
                self._rowspan_carry.pop(col, None)
            else:
                self._rowspan_carry[col] = (remaining - 1, text, is_header)
            col = len(self._row)

    def handle_endtag(self, tag: str) -> None:
        name = (tag or '').casefold()
        if name in {'td', 'th'} and self._cell_parts is not None:
            self._cell_depth -= 1
            if self._cell_depth <= 0 and self._row is not None and self._row_meta is not None:
                self._emit_carried_cells()
                text = collapse(''.join(self._cell_parts))
                rowspan, colspan = self._pending_span
                for _ in range(max(1, colspan)):
                    self._row.append(text)
                    self._row_meta.append({
                        'text': text,
                        'rowspan': rowspan,
                        'colspan': colspan,
                        'is_header': self._pending_is_header,
                        'from_rowspan': False,
                    })
                if rowspan > 1:
                    # Carry into subsequent rows for each spanned column just written.
                    start_col = len(self._row) - max(1, colspan)
                    for col in range(start_col, len(self._row)):
                        self._rowspan_carry[col] = (rowspan - 1, text, self._pending_is_header)
                self._cell_parts = None
                self._cell_depth = 0
                self._pending_span = (1, 1)
                self._pending_is_header = False
        elif name == 'tr' and self._row is not None and self._row_meta is not None:
            if self._cell_parts is not None:
                self._emit_carried_cells()
                text = collapse(''.join(self._cell_parts))
                rowspan, colspan = self._pending_span
                for _ in range(max(1, colspan)):
                    self._row.append(text)
                    self._row_meta.append({
                        'text': text,
                        'rowspan': rowspan,
                        'colspan': colspan,
                        'is_header': self._pending_is_header,
                        'from_rowspan': False,
                    })
                self._cell_parts = None
                self._cell_depth = 0
            self._emit_carried_cells()
            self.rows.append(self._row)
            self.row_metas.append(self._row_meta)
            self._row = None
            self._row_meta = None

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None and data:
            self._cell_parts.append(data)


def _html_table_rows(document_text: str) -> list[list[str]]:
    """Parse HTML tables into row/cell text using html.parser."""
    parser = _HtmlTableCellCollector()
    try:
        parser.feed(document_text or '')
        parser.close()
    except Exception:
        return []
    return parser.rows


def _table_ownership_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load section_aware_evidence.table_ownership_resolver knobs."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    section = data.get('section_aware_evidence') if isinstance(
        data.get('section_aware_evidence'), dict
    ) else {}
    cfg = section.get('table_ownership_resolver') if isinstance(
        section.get('table_ownership_resolver'), dict
    ) else {}
    table = data.get('table_extraction') if isinstance(data.get('table_extraction'), dict) else {}
    subject_markers = [
        collapse(str(x or '')).casefold()
        for x in (table.get('subject_column_markers') or [])
        if collapse(str(x or ''))
    ]
    try:
        max_row = int(cfg.get('max_row_chars', 1200))
    except (TypeError, ValueError):
        max_row = 1200
    try:
        max_group = int(cfg.get('max_group_rows', 24))
    except (TypeError, ValueError):
        max_group = 24
    return {
        'enabled': bool(cfg.get('enabled', True)),
        'prefer_subject_column': bool(cfg.get('prefer_subject_column', True)),
        'accept_unique_row_owner_without_section': bool(
            cfg.get('accept_unique_row_owner_without_section', True)
        ),
        'inherit_owner_on_continuation_rows': bool(
            cfg.get('inherit_owner_on_continuation_rows', True)
        ),
        'reset_owner_on_section_boundary': bool(
            cfg.get('reset_owner_on_section_boundary', True)
        ),
        'attach_owner_cell_provenance': bool(
            cfg.get('attach_owner_cell_provenance', True)
        ),
        'max_row_chars': max(80, min(max_row, 8000)),
        'max_group_rows': max(1, min(max_group, 200)),
        'subject_column_markers': subject_markers,
    }


@dataclass
class TableOwnedRow:
    """One physical table row with resolved (or inherited) owner."""
    kind: str
    start: int
    end: int
    row_text: str
    cells: list[str] = field(default_factory=list)
    owner_surface: str | None = None
    owner_tags: list[str] = field(default_factory=list)
    owner_cell_text: str = ''
    owner_cell_start: int | None = None
    owner_cell_end: int | None = None
    is_continuation: bool = False
    ambiguous: bool = False
    group_id: int = -1
    table_id: int = -1
    subject_col: int = 0


@dataclass
class TableOwnershipIndex:
    """Document-scoped table row ownership with absolute offsets."""
    rows: list[TableOwnedRow] = field(default_factory=list)

    def row_at_offset(self, offset: int) -> TableOwnedRow | None:
        hits = [
            row for row in self.rows
            if row.start <= offset < row.end
        ]
        if not hits:
            return None
        hits.sort(key=lambda r: (r.end - r.start, r.start))
        return hits[0]


def _subject_column_index(
    header_cells: list[str],
    *,
    markers: list[str],
) -> int:
    if not header_cells or not markers:
        return 0
    for idx, cell in enumerate(header_cells):
        folded = collapse(cell or '').casefold()
        if any(marker in folded for marker in markers):
            return idx
    return 0


def _owner_tags_from_cell(cell: str, *, policy: Policy | None = None) -> list[str]:
    return _collapse_equivalent_owner_tags(
        _engineering_tags_in(cell or '', policy=policy),
        policy=policy,
    )


def _is_markdown_heading_line(line: str) -> bool:
    stripped = (line or '').lstrip()
    if stripped.startswith('#'):
        return True
    raw = stripped.strip()
    return bool(len(raw) >= 4 and raw.startswith('**') and raw.endswith('**'))


def _is_pipe_separator_row(line: str) -> bool:
    """True for Markdown pipe alignment rows (| --- | :---: |)."""
    stripped = (line or '').strip()
    if '|' not in stripped:
        return False
    raw = stripped.strip('|')
    cells = [collapse(c) for c in raw.split('|')]
    if len(cells) < 2:
        return False
    return all(c and set(c) <= {'-', ':', ' '} for c in cells)


def _iter_pipe_table_blocks(
    document_text: str,
) -> list[dict[str, Any]]:
    """Group contiguous Markdown pipe-table lines with absolute offsets.

    Separator rows keep the block open so owner + continuation rows stay one table.
    """
    doc = document_text or ''
    blocks: list[dict[str, Any]] = []
    offset = 0
    current: dict[str, Any] | None = None
    for line in doc.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        if _is_pipe_separator_row(line):
            if current is not None:
                current['end'] = offset
            continue
        cells = _split_pipe_row(line)
        if cells is None:
            if current is not None:
                blocks.append(current)
                current = None
            continue
        if current is None:
            current = {
                'kind': 'pipe',
                'start': line_start,
                'end': offset,
                'rows': [],
            }
        current['end'] = offset
        current['rows'].append({
            'start': line_start,
            'end': line_start + len(line.rstrip('\r\n')),
            'text': line.rstrip('\r\n'),
            'cells': cells,
        })
    if current is not None:
        blocks.append(current)
    return blocks


def _iter_html_table_blocks(document_text: str) -> list[dict[str, Any]]:
    """Locate HTML <table> blocks and parse rows with rowspan/colspan + offsets."""
    doc = document_text or ''
    lower = doc.casefold()
    blocks: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = lower.find('<table', pos)
        if start < 0:
            break
        end = lower.find('</table>', start)
        if end < 0:
            break
        end = end + len('</table>')
        chunk = doc[start:end]
        parser = _HtmlTableCellCollector()
        try:
            parser.feed(chunk)
            parser.close()
        except Exception:
            pos = end
            continue
        # Offset-aware <tr> slices inside this table.
        tr_spans: list[tuple[int, int, str]] = []
        local = chunk
        local_lower = local.casefold()
        cursor = 0
        while True:
            tr_start = local_lower.find('<tr', cursor)
            if tr_start < 0:
                break
            tr_end = local_lower.find('</tr>', tr_start)
            if tr_end < 0:
                break
            tr_end = tr_end + len('</tr>')
            tr_spans.append((
                start + tr_start,
                start + tr_end,
                local[tr_start:tr_end],
            ))
            cursor = tr_end
        rows_out: list[dict[str, Any]] = []
        for idx, cells in enumerate(parser.rows):
            meta = parser.row_metas[idx] if idx < len(parser.row_metas) else []
            if idx < len(tr_spans):
                row_start, row_end, row_html = tr_spans[idx]
            else:
                row_start, row_end, row_html = start, end, chunk
            rows_out.append({
                'start': row_start,
                'end': row_end,
                'text': row_html,
                'cells': cells,
                'cell_metas': meta,
            })
        blocks.append({
            'kind': 'html',
            'start': start,
            'end': end,
            'rows': rows_out,
        })
        pos = end
    return blocks


def build_table_ownership_index(
    document_text: str,
    *,
    policy: Policy | None = None,
) -> TableOwnershipIndex:
    """Build once per document: row owners + continuation inheritance."""
    pol = policy if policy is not None else load_policy()
    settings = _table_ownership_settings(pol)
    doc = document_text or ''
    cache_key = (_policy_cache_token(pol), _document_cache_token(doc))
    cached = _DOC_TABLE_OWNER_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = TableOwnershipIndex()
    if not doc or not settings.get('enabled', True):
        _DOC_TABLE_OWNER_INDEX_CACHE[cache_key] = index
        return index

    sections = build_equipment_section_index(doc, policy=pol) if settings.get(
        'reset_owner_on_section_boundary', True
    ) else []
    max_row_chars = int(settings['max_row_chars'])
    max_group_rows = int(settings['max_group_rows'])
    markers = list(settings.get('subject_column_markers') or [])
    prefer_subject = bool(settings.get('prefer_subject_column', True))
    inherit = bool(settings.get('inherit_owner_on_continuation_rows', True))

    blocks = _iter_pipe_table_blocks(doc) + _iter_html_table_blocks(doc)
    blocks.sort(key=lambda b: int(b.get('start') or 0))
    group_id = 0
    table_id = 0

    for block in blocks:
        rows = list(block.get('rows') or [])
        if not rows:
            continue
        kind = collapse(str(block.get('kind') or 'pipe'))
        header_cells = list(rows[0].get('cells') or [])
        subject_col = (
            _subject_column_index(header_cells, markers=markers)
            if prefer_subject else 0
        )
        # Skip pure header rows that have no owner tags.
        start_idx = 0
        if header_cells and not _owner_tags_from_cell(
            ' '.join(header_cells), policy=pol,
        ):
            header_folded = ' '.join(header_cells).casefold()
            if any(m in header_folded for m in markers) or any(
                not _engineering_tags_in(c, policy=pol) for c in header_cells
            ):
                # Treat first row as header only when it looks like labels, not data.
                if not any(_owner_tags_from_cell(c, policy=pol) for c in header_cells):
                    start_idx = 1

        active_owner: str | None = None
        active_tags: list[str] = []
        active_cell = ''
        active_cell_span: tuple[int, int] | None = None
        active_group = -1
        group_len = 0
        active_section_key: tuple[Any, ...] | None = None

        for row in rows[start_idx:]:
            row_start = int(row.get('start') or 0)
            row_end = int(row.get('end') or row_start)
            row_text = str(row.get('text') or doc[row_start:row_end])
            cells = [collapse(str(c or '')) for c in (row.get('cells') or [])]
            if len(row_text) > max_row_chars:
                active_owner = None
                active_tags = []
                active_group = -1
                group_len = 0
                continue

            if settings.get('reset_owner_on_section_boundary', True) and sections:
                sec = section_at_offset(sections, row_start)
                sec_key = (
                    collapse(str((sec or {}).get('section_number') or '')),
                    int((sec or {}).get('heading_start') or -1),
                ) if sec else None
                if active_section_key is not None and sec_key != active_section_key:
                    active_owner = None
                    active_tags = []
                    active_group = -1
                    group_len = 0
                active_section_key = sec_key

            # Subject/identity column only (header markers or leftmost). Never
            # promote description/value-column tags to row owners.
            subject_idx = subject_col if subject_col < len(cells) else 0
            subject_cell = cells[subject_idx] if cells else ''
            owner_tags = _owner_tags_from_cell(subject_cell, policy=pol)
            ambiguous = len(owner_tags) > 1
            is_continuation = False
            owner_surface = None
            owner_cell_start = None
            owner_cell_end = None

            if ambiguous:
                active_owner = None
                active_tags = []
                active_group = -1
                group_len = 0
            elif owner_tags:
                owner_surface = _owner_surface_for_tag(
                    owner_tags[0], row_text=subject_cell or row_text, policy=pol,
                )
                # Prefer full subject cell as owner surface when it contains the tag.
                if subject_cell and _owner_surfaces_match(
                    subject_cell, owner_surface, policy=pol,
                ):
                    owner_surface = subject_cell
                active_owner = owner_surface
                active_tags = [owner_surface]
                active_cell = subject_cell or owner_surface
                # Best-effort owner-cell offsets inside the row slice.
                cell_key = (subject_cell or owner_surface).casefold()
                folded_row = row_text.casefold()
                rel = folded_row.find(cell_key) if cell_key else -1
                if rel >= 0:
                    owner_cell_start = row_start + rel
                    owner_cell_end = owner_cell_start + len(subject_cell or owner_surface)
                    active_cell_span = (owner_cell_start, owner_cell_end)
                else:
                    active_cell_span = None
                group_id += 1
                active_group = group_id
                group_len = 1
            elif inherit and active_owner and group_len < max_group_rows:
                is_continuation = True
                owner_surface = active_owner
                owner_tags = list(active_tags)
                if active_cell_span:
                    owner_cell_start, owner_cell_end = active_cell_span
                subject_cell = active_cell
                group_len += 1
            else:
                active_owner = None
                active_tags = []
                active_group = -1
                group_len = 0

            index.rows.append(TableOwnedRow(
                kind=f'{kind}_row',
                start=row_start,
                end=row_end,
                row_text=row_text,
                cells=cells,
                owner_surface=owner_surface,
                owner_tags=list(owner_tags or ([owner_surface] if owner_surface else [])),
                owner_cell_text=subject_cell if owner_surface else '',
                owner_cell_start=owner_cell_start,
                owner_cell_end=owner_cell_end,
                is_continuation=is_continuation,
                ambiguous=ambiguous,
                group_id=active_group,
                table_id=table_id,
                subject_col=subject_idx,
            ))
        table_id += 1

    index.rows.sort(key=lambda r: r.start)
    _DOC_TABLE_OWNER_INDEX_CACHE[cache_key] = index
    return index


def _table_owner_provenance(
    row: TableOwnedRow,
    *,
    source_file: str | None = None,
    resolver: str = 'table_row_owner',
) -> dict[str, Any]:
    """Owner-cell provenance kept separate from field/value evidence."""
    return {
        'source_text': row.owner_cell_text or row.owner_surface or row.row_text,
        'source_file': source_file,
        'start_char': row.owner_cell_start,
        'end_char': row.owner_cell_end,
        'equipment_tags': list(row.owner_tags or []),
        'kind': 'table_row',
        'resolver': resolver,
        'row_kind': row.kind,
        'is_continuation': bool(row.is_continuation),
        'group_id': row.group_id,
        'table_id': row.table_id,
        'row_start': row.start,
        'row_end': row.end,
    }


def _resolve_attribute_via_table_row(
    document_text: str,
    evidence: dict,
    *,
    subject: str,
    obj: str,
    predicate: str = '',
    source_file: str | None = None,
    policy: Policy | None = None,
    owner_resolver: Callable[..., str | None] | None = None,
) -> tuple[bool, dict | None, tuple[int, int] | None, str | None]:
    """Bind value-only attribute quotes to a unique table row-group owner.

    LLM may suggest owners only when multiple candidate rows remain ambiguous;
    Python still requires explicit cell alignment or unambiguous continuation.
    """
    del predicate  # reserved for future cue-aware table binding
    settings = _table_ownership_settings(policy)
    if not settings.get('enabled', True):
        return False, None, None, None
    if not settings.get('accept_unique_row_owner_without_section', True):
        return False, None, None, None
    quote = str(evidence.get('source_text') or '')
    if not quote or not _quote_value_supported(quote, obj, policy=policy):
        return False, None, None, None

    pol = policy if policy is not None else load_policy()
    index = build_table_ownership_index(document_text, policy=pol)
    if not index.rows:
        return False, None, None, None

    spans = _attribute_evidence_spans(document_text, evidence)
    if not spans:
        return False, None, None, None

    matched: list[tuple[tuple[int, int], TableOwnedRow]] = []
    for start, end in spans:
        row = index.row_at_offset(start)
        if row is None or row.ambiguous or not row.owner_surface:
            continue
        if subject and not _owner_surfaces_match(subject, row.owner_surface, policy=pol):
            # Keep as candidate for correction / LLM disambiguation.
            matched.append(((start, end), row))
            continue
        matched.append(((start, end), row))

    if not matched:
        return False, None, None, None

    # Prefer rows whose owner already matches the claimed subject.
    subject_hits = [
        item for item in matched
        if _owner_surfaces_match(subject, item[1].owner_surface or '', policy=pol)
    ]
    pool = subject_hits or matched
    owners = _collapse_equivalent_owner_tags(
        [str(row.owner_surface or '') for _, row in pool],
        policy=pol,
    )
    if len(owners) == 1:
        span, row = next(
            item for item in pool
            if _owner_surfaces_match(str(item[1].owner_surface or ''), owners[0], policy=pol)
        )
        prov = _table_owner_provenance(
            row,
            source_file=source_file,
            resolver='table_row_group' if row.is_continuation else 'table_row_owner',
        ) if settings.get('attach_owner_cell_provenance', True) else None
        return True, prov, span, row.owner_surface

    # Ambiguous multi-owner: optional LLM suggestion among candidate owners.
    section_settings = _section_aware_evidence_settings(pol)
    allow_llm = bool(section_settings.get('llm_owner_fallback', True))
    only_ambiguous = bool(section_settings.get('llm_owner_fallback_only_when_ambiguous', True))
    if not allow_llm or not only_ambiguous:
        return False, None, None, None
    # Build lightweight candidate sections from table rows for the existing LLM hook.
    candidates = []
    for _, row in pool:
        candidates.append({
            'heading_text': row.owner_surface,
            'title': row.owner_surface,
            'tags': list(row.owner_tags or []),
            'heading_start': row.owner_cell_start if row.owner_cell_start is not None else row.start,
            'heading_end': row.owner_cell_end if row.owner_cell_end is not None else row.end,
            'section_number': f'table-group:{row.group_id}',
        })
    picked = _resolve_owner_via_llm(
        subject=subject,
        obj=obj,
        predicate='',
        quote=quote,
        candidates=candidates,
        owner_resolver=owner_resolver,
    )
    if picked is None:
        return False, None, None, None
    picked_surface = collapse(str(picked.get('heading_text') or ''))
    for span, row in pool:
        if _owner_surfaces_match(picked_surface, row.owner_surface or '', policy=pol):
            # Still require structural uniqueness of the chosen row owner.
            if not row.owner_surface or row.ambiguous:
                return False, None, None, None
            prov = _table_owner_provenance(
                row, source_file=source_file, resolver='table_row_llm_verified',
            )
            return True, prov, span, row.owner_surface
    return False, None, None, None


def _html_adjacent_td_field_labels(document_text: str) -> list[str]:
    """Labels from adjacent HTML table cells (label cell → value cell)."""
    out: list[str] = []
    seen: set[str] = set()
    for row in _html_table_rows(document_text):
        for index in range(len(row) - 1):
            left = row[index]
            right = row[index + 1]
            if not right or not _is_plausible_kv_label(left):
                continue
            key = left.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(left)
    return out


def _looks_like_identity_code_surface(surface: str, *, policy: Policy | None = None) -> bool:
    """True for tag/doc-number style surfaces that must not be treated as field labels."""
    text = collapse(surface or '')
    if not text:
        return False
    if _engineering_tags_in(text, policy=policy):
        return True
    # Letter+digit identity codes (document nos, tags) keep entity-worthiness.
    if re.search(r'[A-Za-z]', text) and re.search(r'\d', text) and re.search(r'[-/_.]', text):
        return True
    return False


def _document_field_label_surfaces(
    document_text: str,
    *,
    policy: Policy | None = None,
) -> set[str]:
    """Casefolded field-label surfaces discovered from source layout (not a word list)."""
    pol = policy if policy is not None else load_policy()
    surfaces: set[str] = set()
    for label in _document_kv_labels(document_text, policy=pol):
        key = collapse(str(label or '')).casefold()
        if key:
            surfaces.add(key)
    for label in _html_adjacent_td_field_labels(document_text):
        key = collapse(label).casefold()
        if key:
            surfaces.add(key)
    return surfaces


def _is_field_label_non_entity(
    surface: str,
    document_text: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """Reject field-label captions using non_entity_policy.reject_field_labels + layout."""
    pol = policy if policy is not None else load_policy()
    nep = non_entity_policy_section(pol)
    if 'reject_field_labels' not in nep:
        return False
    text = collapse(surface or '')
    if not text:
        return False
    if _looks_like_identity_code_surface(text, policy=pol):
        return False
    labels = _document_field_label_surfaces(document_text, policy=pol)
    return text.casefold() in labels


def _strip_label_unit_suffix(label: str) -> str:
    """'Design Temperature (°C)' -> 'Design Temperature' for ontology token overlap."""
    text = collapse(label or '')
    return collapse(re.sub(r'[\(\[][^\)\]]*[\)\]]\s*$', '', text))


def _split_kv_segment(
    segment: str,
    *,
    separators: tuple[str, ...] | None = None,
    policy: Policy | None = None,
) -> tuple[str, str, str] | None:
    """Split one label/value segment. Returns (label, separator, value) or None."""
    text = segment or ''
    if not text.strip():
        return None
    seps = separators or _kv_separators(policy)
    best: tuple[int, str, str, str] | None = None
    for sep in seps:
        idx = 0
        while True:
            pos = text.find(sep, idx)
            if pos < 0:
                break
            left = text[:pos]
            right = text[pos + len(sep):]
            label = collapse(left)
            value = collapse(right.split('\n', 1)[0]) if right else ''
            # Allow unit suffix glued to label before separator.
            if not _is_plausible_kv_label(label):
                idx = pos + 1
                continue
            if not value:
                idx = pos + 1
                continue
            # Prefer earliest split; ties already ordered by longest separator.
            if best is None or pos < best[0]:
                best = (pos, label, sep, value)
            break
    if best is None:
        return None
    return best[1], best[2], best[3]


def _iter_line_kv_rows(
    document_text: str,
    *,
    separators: tuple[str, ...] | None = None,
    policy: Policy | None = None,
) -> list[dict[str, Any]]:
    """Parse Markdown lines and pipe-table rows into KV records with offsets."""
    text = document_text or ''
    seps = separators or _kv_separators(policy)
    rows: list[dict[str, Any]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_text = line.rstrip('\r\n')
        line_start = offset
        offset += len(line)
        stripped = line_text.strip()
        if not stripped or stripped.startswith('#'):
            continue
        cells = _split_pipe_row(line_text)
        if cells is not None:
            label = cells[0]
            value = cells[1]
            if _is_plausible_kv_label(label) and value and set(label) > {'-', ':'}:
                rows.append({
                    'label': label,
                    'value': value,
                    'separator': '|',
                    'source_text': line_text.strip(),
                    'start_char': line_start + line_text.find('|'),
                    'end_char': line_start + len(line_text.rstrip()),
                    'kind': 'pipe',
                })
            continue
        parsed = _split_kv_segment(stripped, separators=seps, policy=policy)
        if not parsed:
            continue
        label, sep, value = parsed
        exact = line_text.find(stripped)
        source = stripped if exact >= 0 else f'{label}{sep}{value}'
        start = line_start + (exact if exact >= 0 else max(0, line_text.find(label)))
        rows.append({
            'label': label,
            'value': value,
            'separator': sep,
            'source_text': source,
            'start_char': start,
            'end_char': start + len(source),
            'kind': 'line',
        })
    return rows


def _find_label_separator_hits(
    text: str,
    labels: list[str],
    separators: tuple[str, ...],
    *,
    policy: Policy | None = None,
) -> list[tuple[int, int, str, str]]:
    """Find (label_start, value_start, label, separator) hits in mashed text."""
    del policy  # reserved for future ontology-aware ranking
    hits: list[tuple[int, int, str, str]] = []
    if not text:
        return hits
    lowered = text.casefold()
    for label in labels:
        base = collapse(label or '')
        if not base:
            continue
        variants = [base, _strip_label_unit_suffix(base)]
        for variant in variants:
            needle = variant.casefold()
            if len(needle) < 2:
                continue
            start_at = 0
            while True:
                idx = lowered.find(needle, start_at)
                if idx < 0:
                    break
                after = text[idx + len(variant):]
                unit = _match_trailing_unit_suffix(after)
                cursor = idx + len(variant) + len(unit)
                matched_sep = ''
                for sep in separators:
                    if text.startswith(sep, cursor):
                        matched_sep = sep
                        break
                if matched_sep:
                    # Avoid mid-token hits.
                    if idx > 0 and (text[idx - 1].isalnum() or text[idx - 1] in '_/'):
                        start_at = idx + 1
                        continue
                    label_text = text[idx:cursor]
                    hits.append((idx, cursor + len(matched_sep), label_text, matched_sep))
                start_at = idx + 1
    hits.sort(key=lambda row: (row[0], -len(row[2])))
    filtered: list[tuple[int, int, str, str]] = []
    last_end = -1
    for hit in hits:
        if hit[0] < last_end:
            continue
        filtered.append(hit)
        last_end = hit[1]
    return filtered


def _iter_concatenated_kv_rows(
    document_text: str,
    *,
    known_labels: list[str],
    separators: tuple[str, ...] | None = None,
    policy: Policy | None = None,
) -> list[dict[str, Any]]:
    """Split HTML/mashed cells into consecutive Label+sep+value fields."""
    text = document_text or ''
    seps = separators or _kv_separators(policy)
    # Seed scanner with discovered + ontology labels; also accept Title-Case runs via line parser.
    labels = [collapse(x) for x in known_labels if collapse(x)]
    # Add labels already visible on their own lines so mashed cells can split.
    for row in _iter_line_kv_rows(text, separators=seps, policy=policy):
        labels.append(str(row.get('label') or ''))
    # Dedupe preserving order.
    seen: set[str] = set()
    uniq_labels: list[str] = []
    for label in labels:
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        uniq_labels.append(label)

    rows: list[dict[str, Any]] = []
    for table in _HTML_TABLE.finditer(text):
        block = table.group(0)
        base = table.start()
        plain_offset_map: list[int] = []
        plain_chars: list[str] = []
        i = 0
        while i < len(block):
            if block[i] == '<':
                end = block.find('>', i)
                if end < 0:
                    break
                i = end + 1
                continue
            plain_chars.append(block[i])
            plain_offset_map.append(base + i)
            i += 1
        plain = ''.join(plain_chars)
        hits = _find_label_separator_hits(plain, uniq_labels, seps, policy=policy)
        for idx, (label_at, value_at, label_text, sep) in enumerate(hits):
            value_end = hits[idx + 1][0] if idx + 1 < len(hits) else len(plain)
            value = collapse(plain[value_at:value_end])
            if not value or not _is_plausible_kv_label(label_text):
                continue
            abs_start = plain_offset_map[label_at]
            abs_end_idx = min(value_end, len(plain_offset_map)) - 1
            abs_end = plain_offset_map[abs_end_idx] + 1 if abs_end_idx >= 0 else abs_start
            source = text[abs_start:abs_end]
            rows.append({
                'label': collapse(label_text),
                'value': value,
                'separator': sep,
                'source_text': source,
                'start_char': abs_start,
                'end_char': abs_end,
                'kind': 'html_concat',
            })
    return rows


def _ontology_boundary_labels(policy: Policy) -> list[str]:
    """Seed KV/heading boundary labels from existing aliases and column markers."""
    data = policy.data if isinstance(policy.data, dict) else {}
    labels: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        label = collapse(raw or '')
        key = label.casefold()
        if not label or key in seen or len(label) < 2:
            return
        seen.add(key)
        labels.append(label)

    assertion = (
        data.get('assertion_classification')
        if isinstance(data.get('assertion_classification'), dict)
        else {}
    )
    aliases = assertion.get('property_concept_aliases')
    if isinstance(aliases, dict):
        for key, value in aliases.items():
            _add(str(value or ''))
            _add(str(key or ''))

    table = data.get('table_extraction') if isinstance(data.get('table_extraction'), dict) else {}
    for key in (
        'measurement_column_markers',
        'quantity_column_markers',
        'description_column_markers',
        'subject_column_markers',
    ):
        for item in table.get(key) or []:
            _add(str(item or ''))
    return labels


def _discover_document_kv_labels(
    document_text: str,
    *,
    seed: list[str] | None = None,
    policy: Policy | None = None,
) -> list[str]:
    """Discover labels from Markdown lines, pipe tables, and concatenated HTML fields."""
    text = document_text or ''
    seps = _kv_separators(policy)
    counts: dict[str, int] = {}
    canonical: dict[str, str] = {}

    def _note(raw: str) -> None:
        label = collapse(raw or '')
        if not _is_plausible_kv_label(label):
            return
        key = label.casefold()
        counts[key] = counts.get(key, 0) + 1
        canonical.setdefault(key, label)
        base = _strip_label_unit_suffix(label)
        if base and base.casefold() != key and _is_plausible_kv_label(base):
            bkey = base.casefold()
            counts[bkey] = counts.get(bkey, 0) + 1
            canonical.setdefault(bkey, base)

    for item in seed or []:
        _note(str(item or ''))

    line_rows = _iter_line_kv_rows(text, separators=seps, policy=policy)
    for row in line_rows:
        _note(str(row.get('label') or ''))

    # Second pass: mashed HTML using labels seen so far + ontology seed.
    for row in _iter_concatenated_kv_rows(
        text,
        known_labels=list(canonical.values()),
        separators=seps,
        policy=policy,
    ):
        _note(str(row.get('label') or ''))

    ranked = sorted(
        canonical.items(),
        key=lambda kv: (-counts.get(kv[0], 0), len(kv[1]), kv[0]),
    )
    return [label for key, label in ranked if counts.get(key, 0) >= 1]


def _document_kv_labels(
    document_text: str,
    *,
    policy: Policy | None = None,
) -> list[str]:
    pol = policy if policy is not None else load_policy()
    settings = _section_aware_evidence_settings(pol)
    cache_key = (_policy_cache_token(pol), _document_cache_token(document_text or ''))
    cached = _DOC_KV_LABEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    seed = list(settings.get('ontology_boundary_labels') or [])
    if settings.get('discover_kv_labels_from_document', True):
        labels = _discover_document_kv_labels(document_text or '', seed=seed, policy=pol)
    else:
        labels = list(seed)
    _DOC_KV_LABEL_CACHE[cache_key] = labels
    return labels


def _section_aware_evidence_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load section_aware_evidence flags; reuse ontology/identifier/table config."""
    pol = policy if policy is not None else load_policy()
    cache_key = _policy_cache_token(pol)
    cached = _SECTION_AWARE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = pol.data if isinstance(pol.data, dict) else {}
    cfg = data.get('section_aware_evidence') if isinstance(data.get('section_aware_evidence'), dict) else {}
    enabled = bool(cfg.get('enabled', True))
    reuse_ids = bool(cfg.get('reuse_identifier_patterns', True))
    reuse_aliases = bool(cfg.get('reuse_property_concept_aliases', True))
    reuse_table = bool(cfg.get('reuse_table_column_markers', True))

    tokens: set[str] = set()
    measurement_tokens: set[str] = set()
    quantity_tokens: set[str] = set()
    ontology_labels: list[str] = []
    table = data.get('table_extraction') if isinstance(data.get('table_extraction'), dict) else {}
    if reuse_aliases or reuse_table:
        full_labels = _ontology_boundary_labels(pol)
        if reuse_aliases and reuse_table:
            ontology_labels = full_labels
        else:
            assertion = (
                data.get('assertion_classification')
                if isinstance(data.get('assertion_classification'), dict)
                else {}
            )
            if reuse_aliases:
                aliases = assertion.get('property_concept_aliases')
                if isinstance(aliases, dict):
                    ontology_labels.extend(collapse(str(v or '')) for v in aliases.values())
                    ontology_labels.extend(collapse(str(k or '')) for k in aliases.keys())
            if reuse_table:
                for key in (
                    'measurement_column_markers',
                    'quantity_column_markers',
                    'description_column_markers',
                    'subject_column_markers',
                ):
                    ontology_labels.extend(collapse(str(x or '')) for x in (table.get(key) or []))
            seen: set[str] = set()
            deduped: list[str] = []
            for label in ontology_labels:
                key = label.casefold()
                if not label or key in seen:
                    continue
                seen.add(key)
                deduped.append(label)
            ontology_labels = deduped
        for label in ontology_labels:
            tokens |= {t for t in _token_parts(label) if len(t) >= 2}
            tokens |= {t for t in _token_parts(_strip_label_unit_suffix(label)) if len(t) >= 2}

    if reuse_table:
        for item in table.get('measurement_column_markers') or []:
            measurement_tokens |= {t for t in _token_parts(item) if len(t) >= 2}
            measurement_tokens |= {
                t for t in _token_parts(_strip_label_unit_suffix(str(item or ''))) if len(t) >= 2
            }
        for item in table.get('quantity_column_markers') or []:
            quantity_tokens |= {t for t in _token_parts(item) if len(t) >= 2}
            quantity_tokens |= {
                t for t in _token_parts(_strip_label_unit_suffix(str(item or ''))) if len(t) >= 2
            }

    tag_pattern = ''
    if reuse_ids:
        identity = data.get('identity_resolution') if isinstance(data.get('identity_resolution'), dict) else {}
        for row in identity.get('identifier_patterns') or []:
            if not isinstance(row, dict):
                continue
            if collapse(str(row.get('kind') or '')).casefold() != 'engineering_tag':
                continue
            tag_pattern = str(row.get('pattern') or '').strip()
            if tag_pattern:
                break
    tag_core = _engineering_tag_core_from_pattern(tag_pattern)

    generic_preds = {
        collapse(p).casefold()
        for p in attribute_predicate_hints(pol)
        if collapse(p)
    }

    settings = {
        'enabled': enabled,
        'reuse_identifier_patterns': reuse_ids,
        'reuse_property_concept_aliases': reuse_aliases,
        'reuse_table_column_markers': reuse_table,
        'discover_kv_labels_from_document': bool(cfg.get('discover_kv_labels_from_document', True)),
        'pint_confirm_measurement_objects': bool(cfg.get('pint_confirm_measurement_objects', True)),
        'llm_owner_fallback': bool(cfg.get('llm_owner_fallback', True)),
        'llm_owner_fallback_only_when_ambiguous': bool(
            cfg.get('llm_owner_fallback_only_when_ambiguous', True)
        ),
        'field_label_tokens': frozenset(tokens),
        'measurement_field_tokens': frozenset(measurement_tokens),
        'quantity_field_tokens': frozenset(quantity_tokens),
        'ontology_boundary_labels': ontology_labels,
        'kv_separators': _kv_separators(pol),
        'engineering_tag_pattern': tag_pattern,
        'engineering_tag_core': tag_core,
        'engineering_tag_re': _compile_regex(tag_pattern) if tag_pattern else None,
        'generic_attribute_predicates': frozenset(generic_preds),
    }
    _SECTION_AWARE_CACHE[cache_key] = settings
    return settings


def _unescape_source_markup(text: str) -> str:
    """Undo simple source escapes so value matching is not markup-sensitive."""
    return collapse(str(text or '')).replace('\\*', '*').replace('\\(', '(').replace('\\)', ')')


def _attribute_value_candidates(obj: str, *, policy: Policy | None = None) -> list[str]:
    """Full object plus value-side after a source label separator, if present."""
    text = collapse(obj or '')
    if not text:
        return []
    candidates = [text]
    parsed = _split_kv_segment(text, separators=_kv_separators(policy))
    if parsed:
        _label, _sep, value = parsed
        if value and value.casefold() != text.casefold():
            candidates.append(value)
    extra: list[str] = []
    for cand in candidates:
        unesc = _unescape_source_markup(cand)
        if unesc and unesc.casefold() != cand.casefold():
            extra.append(unesc)
        starred = cand.replace('*', '\\*')
        if starred != cand:
            extra.append(starred)
        if len(unesc) >= 24:
            prefix = unesc[:24].rstrip()
            if len(prefix) >= 16:
                extra.append(prefix)
    candidates.extend(extra)
    return list(dict.fromkeys(candidates))


def _is_generic_attribute_predicate(
    predicate: str,
    *,
    policy: Policy | None = None,
) -> bool:
    settings = _section_aware_evidence_settings(policy)
    preds = settings.get('generic_attribute_predicates') or frozenset()
    return collapse(predicate or '').casefold() in preds


def _looks_like_labeled_attribute_object(obj: str, *, policy: Policy | None = None) -> bool:
    return _split_kv_segment(obj or '', separators=_kv_separators(policy)) is not None


_PINT_UREG: Any = None
_PINT_IMPORT_TRIED = False


def _pint_registry() -> Any | None:
    """Lazy UnitRegistry; None when pint is not installed."""
    global _PINT_UREG, _PINT_IMPORT_TRIED
    if _PINT_UREG is not None:
        return _PINT_UREG
    if _PINT_IMPORT_TRIED:
        return None
    _PINT_IMPORT_TRIED = True
    try:
        from pint import UnitRegistry
    except ImportError:
        return None
    ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    # Common datasheet spellings not always in the default registry.
    for defn in (
        't = metric_ton',
        'Nm3 = meter ** 3',
        'nm3 = meter ** 3',
    ):
        try:
            ureg.define(defn)
        except Exception:
            pass
    _PINT_UREG = ureg
    return _PINT_UREG


def _normalize_quantity_text(text: str) -> str:
    """Light normalization so Pint can parse datasheet unit spellings."""
    s = collapse(text or '')
    if not s:
        return ''
    replacements = (
        ('℃', ' degC'),
        ('℉', ' degF'),
        ('°C', ' degC'),
        ('°F', ' degF'),
        ('ºC', ' degC'),
        ('ºF', ' degF'),
        ('³', '**3'),
        ('²', '**2'),
        ('µ', 'u'),
        ('×', '*'),
        ('·', '*'),
        ('/', ' / '),
    )
    for old, new in replacements:
        s = s.replace(old, new)
    # "kg/m3" / "kg / m 3" → keep parseable; strip trailing punctuation.
    s = re.sub(r'[,;]+$', '', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def _object_looks_like_numeric_measurement(
    obj: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """True when the object side looks like a number or number+unit (not a tag).

    Letter+digit codes (N2, DN100) are treated as categorical, not quantities.
    """
    settings = _section_aware_evidence_settings(policy)
    tag_re = settings.get('engineering_tag_re')
    # Short codes only (N2, DN100, NB80) — not prose like "approx 90-100ish".
    code_like = re.compile(r'^[A-Za-z]{1,4}\d{1,5}[A-Za-z]{0,3}$')
    for cand in _attribute_value_candidates(obj, policy=policy):
        text = collapse(cand or '')
        if not text or not re.search(r'\d', text):
            continue
        if tag_re is not None and tag_re.fullmatch(text):
            continue
        if _engineering_tags_in(text, policy=policy) and tag_re is not None and tag_re.fullmatch(
            collapse(text)
        ):
            continue
        compact = re.sub(r'[\s\-_/]+', '', text)
        if len(compact) <= 10 and code_like.fullmatch(compact):
            continue
        return True
    return False


def _object_parses_as_quantity(obj: str, *, policy: Policy | None = None) -> bool:
    """Confirm object is a bare number or Pint-parseable quantity/unit literal."""
    bare_number = re.compile(r'^[+-]?\d+(?:[.,]\d+)?(?:\s*[eE][+-]?\d+)?$')
    ureg = _pint_registry()
    for cand in _attribute_value_candidates(obj, policy=policy):
        text = collapse(cand or '')
        if not text:
            continue
        compact = text.replace(',', '')
        if bare_number.fullmatch(compact):
            return True
        normalized = _normalize_quantity_text(compact)
        if not normalized:
            continue
        if bare_number.fullmatch(normalized.replace(' ', '')):
            return True
        if ureg is None:
            # Soft fallback without Pint: digit + unit-ish letter/symbol.
            if re.search(r'\d', normalized) and re.search(r'[A-Za-z%]', normalized):
                return True
            continue
        try:
            ureg.parse_expression(normalized)
            return True
        except Exception:
            # Retry glued unit suffixes only (93t/hr → 93 t / hr).
            # Do not split letter+digit codes (N2, DN100) into unit expressions.
            spaced = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', normalized)
            if spaced != normalized:
                try:
                    ureg.parse_expression(spaced)
                    return True
                except Exception:
                    pass
            continue
    return False


def _engineering_tags_in(
    text: str,
    *,
    policy: Policy | None = None,
) -> list[str]:
    settings = _section_aware_evidence_settings(policy)
    tag_re = settings.get('engineering_tag_re')
    if tag_re is None:
        return []
    out: list[str] = []
    for match in tag_re.finditer(text or ''):
        if 'tag' in match.re.groupindex:
            value = match.group('tag')
        else:
            value = match.group(0)
        if value:
            out.append(value)
    return out


def _normalize_equipment_tag(tag: str) -> str:
    return collapse(tag or '').casefold()


def _tags_equivalent(left: str, right: str) -> bool:
    a = _normalize_equipment_tag(left)
    b = _normalize_equipment_tag(right)
    if not a or not b:
        return False
    if a == b:
        return True
    a_base, b_base = a.split('/', 1)[0], b.split('/', 1)[0]
    return a == b_base or b == a_base or a_base == b_base


def _subject_matches_equipment_tags(
    subject: str,
    tags: list[str],
    *,
    policy: Policy | None = None,
) -> bool:
    subject_tags = _engineering_tags_in(subject, policy=policy)
    if not subject_tags:
        subject_tags = [collapse(subject or '')] if collapse(subject or '') else []
    if not subject_tags or not tags:
        return False
    return any(
        _tags_equivalent(subj_tag, sec_tag)
        for subj_tag in subject_tags
        for sec_tag in tags
    )


def _subject_matches_section(
    subject: str,
    section: dict[str, Any],
    *,
    policy: Policy | None = None,
) -> bool:
    tags = list(section.get('tags') or [])
    if _subject_matches_equipment_tags(subject, tags, policy=policy):
        return True
    surface = collapse(subject or '').casefold()
    if not surface:
        return False
    # Engineering-tag subjects only own via heading tags — not via tags mentioned
    # in a neighbor title (e.g. "22-G-3800: 22-V-3800 Filter").
    if _engineering_tags_in(subject, policy=policy):
        return False
    title = collapse(str(section.get('title') or '')).casefold()
    heading = collapse(str(section.get('heading_text') or '')).casefold()
    return bool(surface in title or surface in heading)


def _collapse_equivalent_owner_tags(
    tags: list[str],
    *,
    policy: Policy | None = None,
) -> list[str]:
    """Deduplicate owner tags under _tags_equivalent (preserve first surface)."""
    out: list[str] = []
    for tag in tags or []:
        surface = collapse(str(tag or ''))
        if not surface:
            continue
        if any(_tags_equivalent(surface, prev) for prev in out):
            continue
        out.append(surface)
    return out


def _owner_surfaces_match(
    left: str,
    right: str,
    *,
    policy: Policy | None = None,
) -> bool:
    a = collapse(left or '')
    b = collapse(right or '')
    if not a or not b:
        return False
    if a.casefold() == b.casefold():
        return True
    left_tags = _engineering_tags_in(a, policy=policy) or [a]
    right_tags = _engineering_tags_in(b, policy=policy) or [b]
    return any(
        _tags_equivalent(lt, rt)
        for lt in left_tags
        for rt in right_tags
    )


def _owner_surface_for_tag(
    tag: str,
    *,
    section: dict[str, Any] | None = None,
    row_text: str = '',
    policy: Policy | None = None,
) -> str:
    """Prefer the tag surface as written in the section heading or attribute row."""
    want = collapse(tag or '')
    if not want:
        return ''
    for text in (
        str((section or {}).get('heading_text') or ''),
        row_text or '',
    ):
        for found in _engineering_tags_in(text, policy=policy):
            if _tags_equivalent(found, want):
                return found
    return want


def _attribute_evidence_spans(
    document_text: str,
    evidence: dict[str, Any],
) -> list[tuple[int, int]]:
    """Locate attribute quote spans; prefer validated supplied offsets."""
    doc = document_text or ''
    quote = str(evidence.get('source_text') or '')
    if not doc or not quote:
        return []
    supplied_start = evidence.get('start_char')
    supplied_end = evidence.get('end_char')
    if (
        isinstance(supplied_start, int)
        and isinstance(supplied_end, int)
        and 0 <= supplied_start < supplied_end <= len(doc)
        and doc[supplied_start:supplied_end] == quote
    ):
        return [(supplied_start, supplied_end)]
    return _find_all_spans(doc, quote)


def _classify_attribute_ownership(
    document_text: str,
    evidence: dict[str, Any],
    *,
    obj: str = '',
    policy: Policy | None = None,
) -> dict[str, Any]:
    """Classify attribute source scope from section heading + boundaries + row.

    Returns scope in {specific, shared, unknown}. A unique structural owner is
    returned only for specific. Shared/general content (outside equipment
    sections, no row owner) never invents a package owner.
    """
    empty = {
        'scope': 'unknown',
        'owner_surface': None,
        'owner_tags': [],
        'section': None,
        'basis': '',
    }
    doc = document_text or ''
    quote = str(evidence.get('source_text') or '')
    if not doc or not quote:
        return empty
    # Value must appear in the quote before structural ownership is trusted.
    if obj and not _quote_value_supported(quote, obj, policy=policy):
        return empty

    pol = policy if policy is not None else load_policy()
    sections = build_equipment_section_index(doc, policy=pol)
    table_index = build_table_ownership_index(doc, policy=pol)
    spans = _attribute_evidence_spans(doc, evidence)
    if not spans:
        return empty

    per_span: list[dict[str, Any]] = []
    for start, end in spans:
        table_row = table_index.row_at_offset(start)
        if table_row is not None:
            row_text = table_row.row_text
            if table_row.ambiguous:
                row_tags = list(table_row.owner_tags or [])
            elif table_row.owner_surface:
                row_tags = [table_row.owner_surface]
            else:
                row_tags = []
        else:
            line_lo, line_hi = _line_span_at(doc, start)
            row_text = doc[line_lo:line_hi]
            row_tags = _collapse_equivalent_owner_tags(
                _engineering_tags_in(row_text, policy=pol), policy=pol,
            )
        quote_slice = doc[start:end]
        quote_tags = _collapse_equivalent_owner_tags(
            _engineering_tags_in(quote_slice, policy=pol), policy=pol,
        )
        section = section_at_offset(sections, start)
        section_tags = _collapse_equivalent_owner_tags(
            list((section or {}).get('tags') or []), policy=pol,
        )
        # Prefer verified table row-group ownership (including continuation rows).
        if (
            table_row is not None
            and not table_row.ambiguous
            and table_row.owner_surface
        ):
            per_span.append({
                'scope': 'specific',
                'owner_surface': table_row.owner_surface,
                'owner_tags': [table_row.owner_surface],
                'section': section,
                'table_row': table_row,
                'basis': (
                    'table_row_group' if table_row.is_continuation else 'table_row'
                ),
            })
            continue
        if table_row is not None and table_row.ambiguous:
            per_span.append({
                'scope': 'unknown',
                'owner_surface': None,
                'owner_tags': list(table_row.owner_tags or []),
                'section': section,
                'table_row': table_row,
                'basis': 'multi_tag_table_row',
            })
            continue
        # Section boundaries own the offset. A row/equipment tag may refine the
        # owner only when that tag appears in the attribute quote itself (so
        # mashed HTML lines with a later inline heading cannot steal earlier
        # fields).
        row_refines = (
            len(quote_tags) == 1
            and (
                section is None
                or len(section_tags) != 1
                or not _tags_equivalent(quote_tags[0], section_tags[0])
            )
        )
        if row_refines:
            owner = _owner_surface_for_tag(
                quote_tags[0], section=section, row_text=row_text, policy=pol,
            )
            per_span.append({
                'scope': 'specific',
                'owner_surface': owner,
                'owner_tags': [owner],
                'section': section,
                'basis': 'attribute_row',
            })
        elif section is not None and len(section_tags) == 1:
            owner = _owner_surface_for_tag(
                section_tags[0], section=section, row_text=row_text, policy=pol,
            )
            per_span.append({
                'scope': 'specific',
                'owner_surface': owner,
                'owner_tags': [owner],
                'section': section,
                'basis': 'equipment_section',
            })
        elif section is not None and len(section_tags) > 1:
            per_span.append({
                'scope': 'unknown',
                'owner_surface': None,
                'owner_tags': section_tags,
                'section': section,
                'basis': 'multi_tag_section',
            })
        elif len(row_tags) == 1:
            owner = _owner_surface_for_tag(
                row_tags[0], section=section, row_text=row_text, policy=pol,
            )
            per_span.append({
                'scope': 'specific',
                'owner_surface': owner,
                'owner_tags': [owner],
                'section': section,
                'basis': 'attribute_row',
            })
        elif len(row_tags) > 1:
            per_span.append({
                'scope': 'unknown',
                'owner_surface': None,
                'owner_tags': row_tags,
                'section': section,
                'basis': 'multi_tag_row',
            })
        else:
            # Outside any equipment section and no row owner → shared/general.
            per_span.append({
                'scope': 'shared',
                'owner_surface': None,
                'owner_tags': [],
                'section': None,
                'basis': 'shared_or_general',
            })

    scopes = {row['scope'] for row in per_span}
    if scopes == {'shared'}:
        return {
            'scope': 'shared',
            'owner_surface': None,
            'owner_tags': [],
            'section': None,
            'basis': 'shared_or_general',
        }
    if scopes == {'specific'}:
        owners = _collapse_equivalent_owner_tags(
            [str(row.get('owner_surface') or '') for row in per_span],
            policy=pol,
        )
        if len(owners) == 1:
            chosen = next(
                row for row in per_span
                if _owner_surfaces_match(
                    str(row.get('owner_surface') or ''), owners[0], policy=pol,
                )
            )
            return {
                'scope': 'specific',
                'owner_surface': owners[0],
                'owner_tags': [owners[0]],
                'section': chosen.get('section'),
                'table_row': chosen.get('table_row'),
                'basis': str(chosen.get('basis') or 'equipment_section'),
            }
    return {
        'scope': 'unknown',
        'owner_surface': None,
        'owner_tags': [],
        'section': None,
        'table_row': None,
        'basis': 'ambiguous_spans',
    }


def _looks_like_technical_attribute_claim(
    predicate: str,
    obj: str,
    *,
    evidence_quote: str = '',
    document_text: str = '',
    policy: Policy | None = None,
) -> bool:
    """True for field-label attributes using ontology + document-discovered labels.

    Numeric/unit objects under measurement/quantity markers are confirmed via Pint
    (or bare number). Non-numeric attributes keep marker/KV/evidence rules.
    Ambiguous numeric values that fail Pint fall through to evidence, else False
    (quarantine path). Hierarchy predicates never count as attributes.
    """
    settings = _section_aware_evidence_settings(policy)
    pred = collapse(predicate or '')
    value = collapse(obj or '')
    if not pred or not value:
        return False
    pol = policy if policy is not None else load_policy()
    if pred.casefold() in hierarchy_predicate_set(pol):
        return False
    if _is_generic_attribute_predicate(pred, policy=policy):
        return True
    tag_re = settings.get('engineering_tag_re')
    if tag_re is not None and tag_re.fullmatch(value):
        return False
    # Object that itself is an engineering identity code is not a literal attribute.
    if _engineering_tags_in(value, policy=policy) and tag_re is not None and tag_re.fullmatch(
        collapse(value)
    ):
        return False
    folded_pred = pred.casefold()
    tokens = {
        t for t in _token_parts(folded_pred)
        if t not in _FIELD_TOKEN_STOPWORDS
    }
    field_tokens = set(settings.get('field_label_tokens') or frozenset())
    measurement_tokens = set(settings.get('measurement_field_tokens') or frozenset())
    quantity_tokens = set(settings.get('quantity_field_tokens') or frozenset())
    if document_text and settings.get('discover_kv_labels_from_document', True):
        for label in _document_kv_labels(document_text, policy=policy):
            field_tokens |= {t for t in _token_parts(label) if len(t) >= 2}
    if tokens & field_tokens:
        meas_qty_hit = bool(tokens & (measurement_tokens | quantity_tokens))
        pint_confirm = bool(settings.get('pint_confirm_measurement_objects', True))
        if (
            pint_confirm
            and meas_qty_hit
            and _object_looks_like_numeric_measurement(value, policy=policy)
        ):
            if _object_parses_as_quantity(value, policy=policy):
                return True
            # Ambiguous number/unit — do not force attribute; try evidence below.
        else:
            # Non-numeric (conveying mode/gas, connection text) or non meas/qty label.
            return True
    if _looks_like_labeled_attribute_object(value, policy=policy):
        return True
    quote = collapse(evidence_quote or '')
    if quote:
        folded_q = quote.casefold()
        pred_base = _strip_label_unit_suffix(pred).casefold()
        if (
            folded_q.startswith(folded_pred)
            or folded_q.startswith(pred_base)
            or f'{folded_pred} :' in folded_q
            or f'{folded_pred}:' in folded_q
            or f'{pred_base} :' in folded_q
            or f'| {folded_pred} |' in folded_q
            or f'| {pred_base} |' in folded_q
        ):
            return any(
                cand.casefold() in folded_q
                for cand in _attribute_value_candidates(value, policy=policy)
                if cand
            )
        # Evidence may be a bare KV line with units / alternate separators.
        parsed = _split_kv_segment(quote, separators=_kv_separators(policy))
        if parsed:
            label, _sep, parsed_value = parsed
            label_tokens = _token_parts(_strip_label_unit_suffix(label))
            if label_tokens & _token_parts(pred_base) and any(
                cand.casefold() in parsed_value.casefold()
                for cand in _attribute_value_candidates(value, policy=policy)
                if cand
            ):
                return True
    return False


def build_equipment_section_index(
    document_text: str,
    *,
    policy: Policy | None = None,
) -> list[dict[str, Any]]:
    """Map active equipment headings across pages, banners, HTML, and pipe tables.

    Uses generic Markdown/HTML/bold structural parsing plus ontology
    identifier_patterns for tags. No document-format regex lists in YAML.
    """
    text = document_text or ''
    if not text:
        return []

    pol = policy if policy is not None else load_policy()
    settings = _section_aware_evidence_settings(pol)
    if not settings.get('enabled', True):
        return []

    cache_key = (_policy_cache_token(pol), _document_cache_token(text))
    cached = _DOC_SECTION_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kv_labels = _document_kv_labels(text, policy=pol)
    candidates: list[dict[str, Any]] = []
    candidates.extend(_iter_atx_markdown_headings(text))
    candidates.extend(_iter_html_headings(text))
    candidates.extend(_iter_bold_headings(text))
    candidates.extend(
        _iter_inline_numbered_equipment_headings(
            text, policy=pol, boundary_labels=kv_labels,
        )
    )

    raw: list[dict[str, Any]] = []
    for row in candidates:
        head = str(row.get('head') or '')
        head_tags = _engineering_tags_in(head, policy=pol)
        if not head_tags:
            continue
        start = int(row['heading_start'])
        end = int(row['heading_end'])
        raw.append({
            'section_number': collapse(str(row.get('section_number') or '')),
            'tags': head_tags,
            'title': collapse(str(row.get('title') or '')),
            'heading_text': text[start:end],
            'heading_start': start,
            'heading_end': end,
            'kind': row.get('kind') or 'markdown',
        })

    raw.sort(key=lambda row: int(row['heading_start']))
    deduped: list[dict[str, Any]] = []
    priority = {'markdown': 0, 'html': 1, 'bold': 2, 'inline': 3}
    for row in raw:
        if deduped and int(deduped[-1]['heading_start']) == int(row['heading_start']):
            prev = deduped[-1]
            if priority.get(str(row.get('kind')), 9) < priority.get(str(prev.get('kind')), 9):
                deduped[-1] = row
            continue
        if row.get('kind') == 'inline' and any(
            int(prev['heading_start']) <= int(row['heading_start']) < int(prev['heading_end'])
            for prev in deduped
        ):
            continue
        deduped.append(row)

    sections: list[dict[str, Any]] = []
    for idx, row in enumerate(deduped):
        start = int(row['heading_start'])
        end = int(deduped[idx + 1]['heading_start']) if idx + 1 < len(deduped) else len(text)
        sections.append({
            **row,
            'start_char': start,
            'end_char': end,
        })
    _DOC_SECTION_INDEX_CACHE[cache_key] = sections
    return sections


def section_at_offset(
    sections: list[dict[str, Any]],
    offset: int,
) -> dict[str, Any] | None:
    if not isinstance(offset, int) or offset < 0:
        return None
    for section in sections:
        start = int(section.get('start_char') or 0)
        end = int(section.get('end_char') or 0)
        if start <= offset < end:
            return section
    return None


def _quote_value_supported(quote: str, obj: str, *, policy: Policy | None = None) -> bool:
    folded = (quote or '').casefold()
    if not folded:
        return False
    unesc = _unescape_source_markup(quote).casefold()
    return any(
        cand.casefold() in folded or cand.casefold() in unesc
        for cand in _attribute_value_candidates(obj, policy=policy)
        if cand
    )


def _endpoint_span_in_quote(
    endpoint: str,
    folded_quote: str,
    *,
    policy: Policy | None = None,
) -> tuple[int, int] | None:
    """Earliest (start, end) span for a grounded endpoint surface/tag in quote."""
    surface = collapse(endpoint or '')
    if not surface or not folded_quote:
        return None
    key = surface.casefold()
    idx = folded_quote.find(key)
    if idx >= 0:
        return (idx, idx + len(key))
    for tag in _engineering_tags_in(surface, policy=policy):
        norm = _normalize_equipment_tag(tag)
        if not norm:
            continue
        idx = folded_quote.find(norm)
        if idx >= 0:
            return (idx, idx + len(norm))
    tokens = [tok.casefold() for tok in _tokenize_surface(surface) if len(tok) >= 6]
    best: tuple[int, int] | None = None
    for tok in tokens:
        idx = folded_quote.find(tok)
        if idx < 0:
            continue
        span = (idx, idx + len(tok))
        if best is None or span[0] < best[0]:
            best = span
    return best


def _predicate_evidence_cues(predicate: str, *, policy: Policy | None = None) -> list[str]:
    """Predicate surface + significant tokens + overlapping YAML predicate markers."""
    pred = collapse(predicate or '')
    if not pred:
        return []
    cues: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        cue = collapse(str(raw or '')).casefold()
        if len(cue) < 3 or cue in seen:
            return
        seen.add(cue)
        cues.append(cue)

    pol = policy if policy is not None else load_policy()
    contract = predicate_contract(pred, policy=pol)
    for cue in _contract_evidence_cues(contract):
        _add(cue)
    canon = canonical_predicate(pred, policy=pol) or pred
    _add(canon)
    pred_tokens = {
        t for t in _token_parts(canon)
        if len(t) >= 4 and t not in _FIELD_TOKEN_STOPWORDS
    }
    for tok in sorted(pred_tokens):
        _add(tok)
    for marker in relation_predicate_hints(pol):
        fold = marker.casefold()
        marker_tokens = _token_parts(fold)
        if canon.casefold() == fold or canon.casefold() in fold or fold in canon.casefold():
            _add(fold)
            continue
        if pred_tokens and (pred_tokens & marker_tokens):
            _add(fold)
    for marker in _membership_evidence_markers(pol):
        marker_tokens = _token_parts(marker)
        if pred_tokens and (pred_tokens & marker_tokens):
            _add(marker)
        elif canon.casefold() in marker or marker in canon.casefold():
            _add(marker)
    return cues


def _predicate_meaning_in_quote(
    quote: str,
    predicate: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """True when quote contains predicate meaning cues (not endpoints alone)."""
    pol = policy if policy is not None else load_policy()
    settings = _claim_evidence_recovery_settings(pol)
    contract = predicate_contract(predicate, policy=pol)
    evidence_cfg = contract.get('evidence') if isinstance(contract.get('evidence'), dict) else {}
    require_meaning = evidence_cfg.get(
        'require_predicate_meaning',
        settings.get('require_predicate_meaning', True),
    )
    if not require_meaning:
        return True
    folded = (quote or '').casefold()
    unesc = _unescape_source_markup(quote or '').casefold()
    if not folded:
        return False
    cues = _predicate_evidence_cues(predicate, policy=pol)
    if not cues:
        return False
    pred = collapse(predicate or '').casefold()
    if pred and pred in hierarchy_predicate_set(pol):
        # Hierarchy also accepts YAML membership markers when predicate tokens are absent.
        if _span_has_membership_evidence(quote, policy=pol):
            return True
    return any(cue in folded or cue in unesc for cue in cues)


def _role_assignment_direction_ok(
    quote: str,
    subject: str,
    obj: str,
    *,
    direction: dict[str, Any],
    policy: Policy | None = None,
) -> bool:
    """YAML role_assignment: verb before/near subject; role particle before object."""
    role = direction.get('role_assignment') if isinstance(
        direction.get('role_assignment'), dict
    ) else {}
    if not role.get('enabled'):
        return False
    folded = (quote or '').casefold()
    if not folded:
        return False
    subj_span = _endpoint_span_in_quote(subject, folded, policy=policy)
    obj_span = _endpoint_span_in_quote(obj, folded, policy=policy)
    if subj_span is None or obj_span is None:
        return False
    verb_cues = [
        collapse(str(v or '')).casefold()
        for v in (role.get('verb_cues') or [])
        if collapse(str(v or ''))
    ]
    particles = [
        collapse(str(p or '')).casefold()
        for p in (role.get('role_particles') or [])
        if collapse(str(p or ''))
    ]
    if not verb_cues:
        return False
    verb_hit: tuple[int, int] | None = None
    for cue in verb_cues:
        idx = folded.find(cue)
        if idx >= 0:
            verb_hit = (idx, idx + len(cue))
            break
    if verb_hit is None:
        return False
    # Verb should not start after the object when object is the role noun.
    if verb_hit[0] > obj_span[0]:
        return False
    if particles:
        particle_ok = False
        for part in particles:
            idx = folded.find(part)
            if idx < 0:
                continue
            # Particle between subject and object (typical "appointed X as Y").
            if subj_span[1] <= idx <= obj_span[0] or obj_span[1] <= idx <= subj_span[0]:
                particle_ok = True
                break
            if idx < obj_span[0] and idx >= verb_hit[0]:
                particle_ok = True
                break
        if not particle_ok:
            return False
    return True


def _claim_direction_supported(
    quote: str,
    subject: str,
    obj: str,
    predicate: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """Predicate cue must support direction for the grounded endpoints."""
    pol = policy if policy is not None else load_policy()
    settings = _claim_evidence_recovery_settings(pol)
    contract = predicate_contract(predicate, policy=pol)
    evidence_cfg = contract.get('evidence') if isinstance(contract.get('evidence'), dict) else {}
    require_direction = evidence_cfg.get(
        'require_claim_direction',
        settings.get('require_claim_direction', True),
    )
    if not require_direction:
        return True
    folded = (quote or '').casefold()
    if not folded:
        return False
    subj_span = _endpoint_span_in_quote(subject, folded, policy=pol)
    obj_span = _endpoint_span_in_quote(obj, folded, policy=pol)
    if subj_span is None or obj_span is None:
        return False
    cues = _predicate_evidence_cues(predicate, policy=pol)
    direction = contract.get('direction') if isinstance(contract.get('direction'), dict) else {}
    if direction.get('kind') == 'symmetric':
        return any(cue in folded for cue in cues) or bool(
            collapse(predicate or '').casefold() in hierarchy_predicate_set(pol)
            and _span_has_membership_evidence(quote, policy=pol)
        )
    if _role_assignment_direction_ok(
        quote, subject, obj, direction=direction, policy=pol,
    ):
        return True
    # Same surface both ends (or identical first hit): cue may sit beside one mention.
    if (
        collapse(subject).casefold() == collapse(obj).casefold()
        or subj_span == obj_span
    ):
        if any(cue in folded for cue in cues):
            return True
        pred = collapse(predicate or '').casefold()
        return bool(
            pred
            and pred in hierarchy_predicate_set(pol)
            and _span_has_membership_evidence(quote, policy=pol)
        )
    window_mode = collapse(str(direction.get('direction_window') or 'between_endpoints')).casefold()
    allow_before = bool(direction.get('allow_cue_before_subject'))
    lo = min(subj_span[0], obj_span[0])
    hi = max(subj_span[1], obj_span[1])
    if window_mode == 'covering_span' or allow_before:
        for cue in cues:
            idx = folded.find(cue)
            if idx >= 0:
                lo = min(lo, idx)
                hi = max(hi, idx + len(cue))
    window = folded[lo:hi]
    if any(cue in window for cue in cues):
        return True
    pred = collapse(predicate or '').casefold()
    if pred and pred in hierarchy_predicate_set(pol):
        return _span_has_membership_evidence(window, policy=pol)
    return False


def _is_incomplete_attribute_action_value(
    obj: str,
    quote: str,
    predicate: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """True when object is a truncated modal/action fragment, not a real attribute value."""
    settings = _claim_evidence_recovery_settings(policy)
    if not settings.get('reject_incomplete_attribute_action_phrases', True):
        return False
    value = collapse(obj or '')
    q = collapse(quote or '')
    pred = collapse(predicate or '')
    if not value or not q or not pred:
        return False
    cues = settings.get('incomplete_attribute_action_cues') or []
    folded_v = value.casefold()
    if not any(cue in folded_v for cue in cues):
        return False
    folded_q = q.casefold()
    idx = folded_q.find(folded_v)
    if idx < 0:
        return False
    after = folded_q[idx + len(folded_v) :]
    pred_tokens = [
        t for t in _token_parts(pred)
        if len(t) >= 4 and t not in _FIELD_TOKEN_STOPWORDS
    ]
    # Agent/action fragment precedes the predicate noun in the source sentence.
    return bool(pred_tokens) and any(tok in after for tok in pred_tokens)


def _attribute_owner_in_quote(
    subject: str,
    quote: str,
    *,
    policy: Policy | None = None,
) -> bool:
    """Attribute owner must appear as full surface or engineering tag — not weak tokens.

    Prevents false verifies like subject token 'pellets' matching value text
    'Polypropylene Pellets' without structural table/section ownership.
    """
    surface = collapse(subject or '')
    folded = (quote or '').casefold()
    if not surface or not folded:
        return False
    if surface.casefold() in folded:
        return True
    for tag in _engineering_tags_in(surface, policy=policy):
        norm = _normalize_equipment_tag(tag)
        if norm and norm in folded:
            return True
    return False


def _claim_supported_by_quote(
    quote: str,
    subject: str,
    obj: str,
    *,
    attribute_mode: bool = False,
    predicate: str = '',
    policy: Policy | None = None,
) -> bool:
    """Evidence must ground subject, predicate meaning, object/value, and direction.

    Verbatim co-occurrence of endpoints alone is not enough for non-attribute claims.
    """
    q = quote or ''
    if not q:
        return False
    folded = q.casefold()
    if attribute_mode:
        # Owner-in-quote is strict for attributes; value-only quotes rely on
        # table/section ownership resolvers in _attach_claim_evidence.
        if not _attribute_owner_in_quote(subject, q, policy=policy):
            return False
        if _is_incomplete_attribute_action_value(
            obj, q, predicate, policy=policy,
        ):
            return False
        return _quote_value_supported(q, obj, policy=policy)
    if not _endpoint_supported_in_quote(subject, folded, policy=policy):
        return False
    if not _endpoint_supported_in_quote(obj, folded, policy=policy):
        return False
    if predicate:
        if not _predicate_meaning_in_quote(q, predicate, policy=policy):
            return False
        if not _claim_direction_supported(
            q, subject, obj, predicate, policy=policy,
        ):
            return False
    return True


def _section_context_provenance(
    section: dict[str, Any],
    *,
    source_file: str | None = None,
    resolver: str = 'deterministic',
) -> dict[str, Any]:
    """Separate provenance for the owning equipment heading (never merged into value quote)."""
    heading = str(section.get('heading_text') or '')
    return {
        'source_text': heading,
        'source_file': source_file,
        'start_char': section.get('heading_start'),
        'end_char': section.get('heading_end'),
        'equipment_tags': list(section.get('tags') or []),
        'section_number': section.get('section_number') or '',
        'kind': section.get('kind') or '',
        'resolver': resolver,
    }


def _format_section_candidate(section: dict[str, Any]) -> str:
    tags = ', '.join(str(t) for t in (section.get('tags') or []))
    return (
        f"section_number={section.get('section_number') or ''} "
        f"tags=[{tags}] title={collapse(str(section.get('title') or ''))!r} "
        f"heading={collapse(str(section.get('heading_text') or ''))!r}"
    )


def _pick_section_from_resolver_reply(
    reply: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    text = collapse(reply or '')
    if not text or text.casefold() in {'none', 'null', 'unknown', 'ambiguous'}:
        return None
    folded = text.casefold()
    for section in candidates:
        num = collapse(str(section.get('section_number') or ''))
        if num and num.casefold() in folded:
            return section
    for section in candidates:
        for tag in section.get('tags') or []:
            if _normalize_equipment_tag(str(tag)) and _normalize_equipment_tag(str(tag)) in folded:
                return section
    return None


def _default_llm_owner_resolver(payload: dict[str, Any]) -> str | None:
    """Ask Bedrock which candidate section owns an ambiguous attribute claim."""
    candidates = payload.get('candidates') or []
    if not candidates:
        return None
    lines = [
        'Pick the single equipment section that owns this attribute claim.',
        'Reply with ONLY the section_number if present, else one equipment tag, else NONE.',
        'Do not invent evidence or rewrite the quote.',
        f"subject: {payload.get('subject')}",
        f"predicate: {payload.get('predicate')}",
        f"object: {payload.get('object')}",
        f"evidence_quote: {payload.get('quote')}",
        'candidates:',
    ]
    for section in candidates:
        lines.append(f'- {_format_section_candidate(section)}')
    prompt = '\n'.join(lines)
    try:
        provider = BedrockProvider()
        if not provider.is_available():
            return None
        return collapse(provider.generate(prompt, max_tokens=64, temperature=0.0))
    except Exception:
        return None


def _resolve_owner_via_llm(
    *,
    subject: str,
    obj: str,
    predicate: str,
    quote: str,
    candidates: list[dict[str, Any]],
    owner_resolver: Callable[..., str | None] | None = None,
) -> dict[str, Any] | None:
    if len(candidates) < 2:
        return None
    # Prefer explicit/ hook resolvers. Do not auto-call Bedrock from unit helpers.
    resolver = owner_resolver if owner_resolver is not None else _OWNER_RESOLVER_HOOK
    if resolver is None:
        return None
    payload = {
        'subject': subject,
        'object': obj,
        'predicate': predicate,
        'quote': quote,
        'candidates': candidates,
    }
    try:
        reply = resolver(payload)
    except Exception:
        return None
    return _pick_section_from_resolver_reply(str(reply or ''), candidates)


def _resolve_attribute_via_section(
    document_text: str,
    evidence: dict,
    *,
    subject: str,
    obj: str,
    predicate: str = '',
    source_file: str | None = None,
    policy: Policy | None = None,
    owner_resolver: Callable[..., str | None] | None = None,
) -> tuple[bool, dict | None, tuple[int, int] | None]:
    """Bind a value-only quote to the active equipment section that owns subject."""
    quote = str(evidence.get('source_text') or '')
    if not quote or not _quote_value_supported(quote, obj, policy=policy):
        return False, None, None

    settings = _section_aware_evidence_settings(policy)
    if not settings.get('enabled', True):
        return False, None, None

    sections = build_equipment_section_index(document_text, policy=policy)
    if not sections:
        return False, None, None

    supplied_start = evidence.get('start_char')
    supplied_end = evidence.get('end_char')
    if (
        isinstance(supplied_start, int)
        and isinstance(supplied_end, int)
        and 0 <= supplied_start < supplied_end <= len(document_text or '')
        and (document_text or '')[supplied_start:supplied_end] == quote
    ):
        spans = [(supplied_start, supplied_end)]
    else:
        spans = _find_all_spans(document_text or '', quote)

    matched: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for start, end in spans:
        section = section_at_offset(sections, start)
        if section and _subject_matches_section(subject, section, policy=policy):
            matched.append(((start, end), section))

    if not matched:
        return False, None, None

    distinct_sections = []
    seen_section_keys: set[tuple[Any, ...]] = set()
    for span, sec in matched:
        key = (
            collapse(str(sec.get('section_number') or '')),
            int(sec.get('heading_start') or -1),
            tuple(_normalize_equipment_tag(t) for t in (sec.get('tags') or [])),
        )
        if key in seen_section_keys:
            continue
        seen_section_keys.add(key)
        distinct_sections.append((span, sec))

    if len(distinct_sections) == 1:
        span, section = distinct_sections[0]
        return True, _section_context_provenance(
            section, source_file=source_file, resolver='deterministic',
        ), span

    # Ambiguous: same value quote under multiple subject-matching section headings.
    allow_llm = bool(settings.get('llm_owner_fallback', True))
    only_ambiguous = bool(settings.get('llm_owner_fallback_only_when_ambiguous', True))
    if not allow_llm or (only_ambiguous and len(distinct_sections) <= 1):
        return False, None, None

    uniq = [sec for _, sec in distinct_sections]
    picked = _resolve_owner_via_llm(
        subject=subject,
        obj=obj,
        predicate=predicate,
        quote=quote,
        candidates=uniq,
        owner_resolver=owner_resolver,
    )
    if picked is None or not _subject_matches_section(subject, picked, policy=policy):
        return False, None, None
    for span, sec in distinct_sections:
        if sec is picked or (
            int(sec.get('heading_start') or -1) == int(picked.get('heading_start') or -2)
            and collapse(str(sec.get('section_number') or ''))
            == collapse(str(picked.get('section_number') or ''))
        ):
            return True, _section_context_provenance(
                picked, source_file=source_file, resolver='llm_fallback',
            ), span
    return False, None, None


def _attach_verified_evidence(meta: dict, document_text: str, *, source_file: str | None = None) -> dict:
    out = dict(meta or {})
    verified = _verify_evidence(out.get('evidence'), document_text, source_file=source_file)
    if verified:
        out['evidence'] = verified
    else:
        out.pop('evidence', None)
    return out


def _attach_claim_evidence(
    meta: dict,
    document_text: str,
    *,
    subject: str,
    obj: str,
    predicate: str = '',
    attribute_mode: bool = False,
    source_file: str | None = None,
    policy: Policy | None = None,
    owner_resolver: Callable[..., str | None] | None = None,
) -> dict:
    """Verify quote offsets, claim support, and weak technical predicates.

    Attribute claims may be grounded by a value span under an active equipment
    section whose heading (elsewhere in the same section / prior page) owns the
    subject. Value evidence and section-context provenance stay separate.
    """
    out = _attach_verified_evidence(meta, document_text, source_file=source_file)
    evidence = out.get('evidence')
    quote = str((evidence or {}).get('source_text') or '') if isinstance(evidence, dict) else ''
    pol = policy if policy is not None else load_policy()
    pred_key = collapse(predicate or '').casefold()
    if pred_key and pred_key in hierarchy_predicate_set(pol):
        # Hierarchy claims must use endpoint/membership routing, never attribute mode.
        attr_mode = False
    else:
        attr_mode = bool(
            attribute_mode
            or _looks_like_technical_attribute_claim(
                predicate,
                obj,
                evidence_quote=quote,
                document_text=document_text,
                policy=policy,
            )
        )
    weak_generic = bool(
        attr_mode
        and _is_generic_attribute_predicate(predicate, policy=policy)
        and _looks_like_labeled_attribute_object(obj, policy=policy)
    )
    if weak_generic:
        out['predicate_quality'] = 'weak_generic'
    elif attr_mode and _is_generic_attribute_predicate(predicate, policy=policy):
        out['predicate_quality'] = 'generic'
    elif predicate:
        out['predicate_quality'] = 'specific'

    out.pop('section_context', None)

    if not isinstance(evidence, dict) or not quote:
        # Deterministic recovery when LLM omitted evidence or quote was stripped.
        recovered = _recover_claim_evidence_quote(
            document_text,
            subject=subject,
            obj=obj,
            predicate=predicate,
            attribute_mode=attr_mode,
            policy=policy,
        )
        if recovered:
            verified = _verify_evidence(
                {'source_text': recovered, 'source_file': source_file},
                document_text,
                source_file=source_file,
            )
            if verified:
                out['evidence'] = verified
                evidence = verified
                quote = str(verified.get('source_text') or '')
                out['evidence_recovery'] = 'deterministic_span'
            else:
                out['evidence_status'] = 'missing'
                out['graph_eligible'] = False
                return out
        else:
            out['evidence_status'] = 'missing'
            out['graph_eligible'] = False
            return out
    else:
        evidence = out.get('evidence')
        quote = str((evidence or {}).get('source_text') or '') if isinstance(evidence, dict) else ''

    if not isinstance(evidence, dict):
        out['evidence_status'] = 'missing'
        out['graph_eligible'] = False
        return out

    # Structural ownership (section heading + boundaries + attribute row) is the
    # source of truth for attribute subjects — not LLM guesses or page banners.
    effective_subject = subject
    ownership: dict[str, Any] | None = None
    if attr_mode and quote:
        ownership = _classify_attribute_ownership(
            document_text,
            evidence if isinstance(evidence, dict) else {'source_text': quote},
            obj=obj,
            policy=policy,
        )
        out['attribute_source_scope'] = ownership.get('scope') or 'unknown'
        proven_owner = collapse(str(ownership.get('owner_surface') or ''))
        if ownership.get('scope') == 'specific' and proven_owner:
            if not _owner_surfaces_match(subject, proven_owner, policy=policy):
                out['original_subject'] = subject
                out['corrected_subject'] = proven_owner
                out['owner_repair'] = {
                    'from': subject,
                    'to': proven_owner,
                    'reason': 'unique_structural_owner',
                    'basis': ownership.get('basis') or 'equipment_section',
                    'resolver': 'deterministic_section_row',
                }
                effective_subject = proven_owner
        elif ownership.get('scope') == 'shared':
            # Shared/general sections: never pin to one package; never fan-out.
            out['original_subject'] = subject

    supported = _claim_supported_by_quote(
        quote,
        effective_subject,
        obj,
        attribute_mode=attr_mode,
        predicate=predicate,
        policy=policy,
    )
    section_context = None
    resolved_span = None
    # Verified table row-group ownership: value quote need not repeat the owner.
    if (
        not supported
        and attr_mode
        and isinstance(ownership, dict)
        and ownership.get('scope') == 'specific'
        and collapse(str(ownership.get('basis') or '')).startswith('table_row')
        and _quote_value_supported(quote, obj, policy=policy)
        and not _is_incomplete_attribute_action_value(
            obj, quote, predicate, policy=policy,
        )
    ):
        table_row = ownership.get('table_row')
        if isinstance(table_row, TableOwnedRow) and table_row.owner_surface:
            supported = True
            if _table_ownership_settings(policy).get('attach_owner_cell_provenance', True):
                section_context = _table_owner_provenance(
                    table_row,
                    source_file=source_file,
                    resolver=(
                        'table_row_group'
                        if table_row.is_continuation
                        else 'table_row_owner'
                    ),
                )
            spans = _attribute_evidence_spans(
                document_text,
                evidence if isinstance(evidence, dict) else {'source_text': quote},
            )
            if len(spans) == 1:
                resolved_span = spans[0]
    if not supported and attr_mode and out.get('attribute_source_scope') != 'shared':
        ok, table_ctx, table_span, table_owner = _resolve_attribute_via_table_row(
            document_text,
            evidence,
            subject=effective_subject,
            obj=obj,
            predicate=predicate,
            source_file=source_file,
            policy=policy,
            owner_resolver=owner_resolver,
        )
        if ok:
            supported = True
            section_context = table_ctx
            resolved_span = table_span
            out['attribute_source_scope'] = 'specific'
            if table_owner and not _owner_surfaces_match(
                effective_subject, table_owner, policy=policy,
            ):
                out['original_subject'] = subject
                out['corrected_subject'] = table_owner
                out['owner_repair'] = {
                    'from': subject,
                    'to': table_owner,
                    'reason': 'unique_table_row_owner',
                    'basis': 'table_row',
                    'resolver': 'table_ownership_resolver',
                }
                effective_subject = table_owner
    if not supported and attr_mode and out.get('attribute_source_scope') != 'shared':
        supported, section_context, resolved_span = _resolve_attribute_via_section(
            document_text,
            evidence,
            subject=effective_subject,
            obj=obj,
            predicate=predicate,
            source_file=source_file,
            policy=policy,
            owner_resolver=owner_resolver,
        )
    if not supported:
        # Do not widen a value-only attribute quote into a subject+value nearby
        # span: mashed HTML can co-mention another section's owner and leak.
        # Shared/general attributes also skip nearby recovery (banner leakage).
        allow_recover = not (
            attr_mode
            and quote
            and (
                _quote_value_supported(quote, obj, policy=policy)
                or out.get('attribute_source_scope') == 'shared'
            )
        )
        recovered = ''
        if allow_recover:
            recovered = _recover_claim_evidence_quote(
                document_text,
                subject=effective_subject,
                obj=obj,
                predicate=predicate,
                attribute_mode=attr_mode,
                policy=policy,
            )
        if recovered and recovered.casefold() != quote.casefold():
            verified = _verify_evidence(
                {'source_text': recovered, 'source_file': source_file},
                document_text,
                source_file=source_file,
            )
            if verified:
                quote2 = str(verified.get('source_text') or '')
                if attr_mode:
                    ok, section_context, resolved_span = _resolve_attribute_via_section(
                        document_text,
                        verified,
                        subject=effective_subject,
                        obj=obj,
                        predicate=predicate,
                        source_file=source_file,
                        policy=policy,
                        owner_resolver=owner_resolver,
                    )
                    if ok:
                        evidence = verified
                        quote = quote2
                        supported = True
                        out['evidence_recovery'] = 'deterministic_span_section'
                elif _claim_supported_by_quote(
                    quote2,
                    effective_subject,
                    obj,
                    attribute_mode=False,
                    predicate=predicate,
                    policy=policy,
                ):
                    evidence = verified
                    quote = quote2
                    supported = True
                    out['evidence_recovery'] = 'deterministic_span_replace'

    # Shared/general scope never becomes graph-eligible under a package owner.
    if attr_mode and out.get('attribute_source_scope') == 'shared':
        supported = False

    evidence = dict(evidence)
    # Disambiguate ambiguous value quotes via section ownership without fabricating text.
    if (
        supported
        and resolved_span is not None
        and evidence.get('start_char') is None
        and evidence.get('end_char') is None
    ):
        evidence['start_char'] = resolved_span[0]
        evidence['end_char'] = resolved_span[1]
    evidence['claim_verified'] = supported
    if not supported:
        if attr_mode and out.get('attribute_source_scope') == 'shared':
            evidence['verification_note'] = (
                'shared_or_general_section_no_unique_owner'
            )
        elif attr_mode and _is_incomplete_attribute_action_value(
            obj, quote, predicate, policy=policy,
        ):
            evidence['verification_note'] = 'incomplete_attribute_action_phrase'
        elif attr_mode:
            evidence['verification_note'] = (
                'quote does not support attribute owner/value'
            )
        elif not _predicate_meaning_in_quote(quote, predicate, policy=policy):
            evidence['verification_note'] = 'quote lacks predicate meaning'
        elif not _claim_direction_supported(
            quote, effective_subject, obj, predicate, policy=policy,
        ):
            evidence['verification_note'] = 'quote does not support claim direction'
        else:
            evidence['verification_note'] = (
                'quote does not support claim endpoints/direction'
            )
    else:
        evidence.pop('verification_note', None)
    if weak_generic:
        evidence['predicate_note'] = (
            'generic predicate with labeled value; prefer field-label predicate '
            'and value-only object'
        )
    out['evidence'] = evidence
    if section_context:
        out['section_context'] = section_context
    elif (
        supported
        and attr_mode
        and isinstance(ownership, dict)
        and out.get('attribute_source_scope') == 'specific'
    ):
        table_row = ownership.get('table_row')
        if isinstance(table_row, TableOwnedRow) and table_row.owner_surface:
            out['section_context'] = _table_owner_provenance(
                table_row,
                source_file=source_file,
                resolver=(
                    'table_row_group'
                    if table_row.is_continuation
                    else 'table_row_owner'
                ),
            )
        elif ownership.get('section'):
            out['section_context'] = _section_context_provenance(
                ownership['section'],
                source_file=source_file,
                resolver='deterministic_section_row',
            )
    out['evidence_status'] = 'verified' if supported else 'unverified'
    # GraphRAG-safe: require verified evidence and non-weak technical predicates.
    out['graph_eligible'] = bool(supported) and not weak_generic
    return out

# ---------------------------------------------------------------------------
# 8) Typed merge — restore LLM rows onto Semantica entities/relations/triplets
# ---------------------------------------------------------------------------
def _typed_rows(payload: Any, key: str) -> list[dict]:
    rows_out: list[dict] = []
    for item in _flatten_typed_payloads(payload):
        if hasattr(item, 'model_dump'):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = item
        else:
            continue
        rows = data.get(key) if isinstance(data, dict) else None
        if isinstance(rows, list):
            rows_out.extend(r for r in rows if isinstance(r, dict))
    return rows_out


def _merge_entities_from_typed(entities: list, typed_payload: Any, document_text: str, *, source_file: str | None = None) -> list:
    rows = _typed_rows(typed_payload, 'entities')
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        key = collapse(str(row.get('text') or '')).casefold()
        if key:
            buckets.setdefault(key, []).append(row.get('metadata') if isinstance(row.get('metadata'), dict) else {})
    for ent in entities or []:
        key = collapse(str(getattr(ent, 'text', '') or '')).casefold()
        queue = buckets.get(key) or []
        llm_meta = queue.pop(0) if queue else {}
        merged = _merge_preserved_metadata(getattr(ent, 'metadata', None) or {}, llm_meta)
        ent.metadata = _attach_verified_evidence(merged, document_text, source_file=source_file)
    return entities


_PLACEHOLDER_ENTITY_TEXTS = frozenset({
    'chunk_text',
    '<<chunk_text>>',
    'text',
    'entity name',
    'another entity',
})


def _ground_entities_in_source(
    entities: list,
    document_text: str,
    *,
    policy: Policy | None = None,
) -> list:
    """Drop placeholder / ungrounded / field-label surfaces before graph merge."""
    pol = policy if policy is not None else load_policy()
    doc_fold = (document_text or '').casefold()
    kept: list = []
    dropped: list[str] = []
    for ent in entities or []:
        surface = collapse(str(getattr(ent, 'text', '') or ''))
        if not surface:
            dropped.append('empty_surface')
            continue
        if surface.casefold() in _PLACEHOLDER_ENTITY_TEXTS:
            dropped.append(f'placeholder:{surface}')
            continue
        if surface.casefold() not in doc_fold:
            dropped.append(f'ungrounded:{surface}')
            continue
        if _is_field_label_non_entity(surface, document_text, policy=pol):
            dropped.append(f'field_label:{surface}')
            continue
        kept.append(ent)
    _diag_record(
        'ground_entities',
        input=len(entities or []),
        kept=len(kept),
        dropped_count=len(dropped),
        dropped=dropped,
    )
    return kept


def _merge_relations_from_typed(relations: list, typed_payload: Any, document_text: str, *, source_file: str | None = None) -> list:
    rows = _typed_rows(typed_payload, 'relations')
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        key = '|'.join([
            collapse(str(row.get('subject') or '')).casefold(),
            collapse(str(row.get('predicate') or '')).casefold(),
            collapse(str(row.get('object') or '')).casefold(),
        ])
        buckets.setdefault(key, []).append(row.get('metadata') if isinstance(row.get('metadata'), dict) else {})
    for rel in relations or []:
        subject = endpoint_text(getattr(rel, 'subject', None))
        predicate = collapse(str(getattr(rel, 'predicate', '') or ''))
        obj = endpoint_text(getattr(rel, 'object', None))
        key = '|'.join([subject.casefold(), predicate.casefold(), obj.casefold()])
        queue = buckets.get(key) or []
        llm_meta = queue.pop(0) if queue else {}
        merged = _merge_preserved_metadata(getattr(rel, 'metadata', None) or {}, llm_meta)
        rel.metadata = _attach_verified_evidence(merged, document_text, source_file=source_file)
    return relations


def _merge_triplets_from_typed(triplets: list, typed_payload: Any, document_text: str, *, source_file: str | None = None) -> list:
    rows = _typed_rows(typed_payload, 'triplets')
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        key = '|'.join([
            collapse(str(row.get('subject') or '')).casefold(),
            collapse(str(row.get('predicate') or '')).casefold(),
            collapse(str(row.get('object') or '')).casefold(),
        ])
        buckets.setdefault(key, []).append(row.get('metadata') if isinstance(row.get('metadata'), dict) else {})
    for trip in triplets or []:
        key = '|'.join([
            collapse(str(getattr(trip, 'subject', '') or '')).casefold(),
            collapse(str(getattr(trip, 'predicate', '') or '')).casefold(),
            collapse(str(getattr(trip, 'object', '') or '')).casefold(),
        ])
        queue = buckets.get(key) or []
        llm_meta = queue.pop(0) if queue else {}
        merged = _merge_preserved_metadata(getattr(trip, 'metadata', None) or {}, llm_meta)
        trip.metadata = _attach_verified_evidence(merged, document_text, source_file=source_file)
    return triplets

# ---------------------------------------------------------------------------
# 9) Guided Semantica hooks — inject unified_extraction into extract_*_llm
# ---------------------------------------------------------------------------
_GUIDED_HOOKS_INSTALLED = False

def _chunk_text_from_semantica_prompt(prompt: str) -> str | None:
    """Pull the current source chunk from Semantica's default LLM prompt."""
    if not prompt:
        return None
    markers = ('Text to extract from:\n', 'Text to extract from:')
    for marker in markers:
        idx = prompt.find(marker)
        if idx < 0:
            continue
        rest = prompt[idx + len(marker) :]
        for stop in ('\nEntities found in text:', '\nKnown entities', '\nRelations found'):
            cut = rest.find(stop)
            if cut >= 0:
                rest = rest[:cut]
                break
        return rest.strip()
    return None

_NER_CHUNK_TEXT = threading.local()


def _fill_guidance_chunk(guidance: str, prompt: str) -> str:
    """Replace chunk placeholder with Semantica's per-call source text (supports chunking)."""
    out = (guidance or '').strip()
    if not out:
        return prompt
    chunk = _chunk_text_from_semantica_prompt(prompt)
    if chunk is None:
        chunk = getattr(_NER_CHUNK_TEXT, 'value', None)
    if chunk is None:
        return out
    # NER-focused guidance: de-emphasize repeated page banners in prompt context only.
    if 'TASK = ENTITIES' in out:
        chunk = _ner_context_text(chunk)
    if '<<CHUNK_TEXT>>' in out:
        return out.replace('<<CHUNK_TEXT>>', chunk)
    if '<<TEXT>>' in out:
        return out.replace('<<TEXT>>', chunk)
    return out


def _attach_domain_instructions(
    llm,
    domain_instructions: str,
    *,
    markers: tuple[str, ...],
    typed_store: dict[str, Any] | None = None,
) -> None:
    guidance = (domain_instructions or '').strip()
    if not guidance or llm is None:
        return
    llm._semantica_domain_guidance = guidance  # type: ignore[attr-defined]
    llm._semantica_domain_markers = markers  # type: ignore[attr-defined]
    call_store = typed_store if typed_store is not None else {}
    llm._semantica_typed_store = call_store  # type: ignore[attr-defined]

    def _augment(prompt: str) -> str:
        guidance = getattr(llm, '_semantica_domain_guidance', '')
        if guidance and guidance.strip():
            return _fill_guidance_chunk(guidance, prompt)
        return prompt

    def _needs_augment(prompt: str) -> bool:
        # generate_typed already rewrites Semantica's prompt; generate_structured
        # must not rebuild from the template (that reintroduces <<CHUNK_TEXT>>).
        text = prompt or ''
        if '<<CHUNK_TEXT>>' in text or '<<TEXT>>' in text:
            return True
        if 'Text to extract from:' in text:
            return True
        return False

    if not getattr(llm, '_semantica_domain_guidance_hooked', False):
        if hasattr(llm, 'generate_typed'):
            original_typed = llm.generate_typed

            def generate_typed(prompt: str, schema=None, **kwargs):
                augmented = _augment(prompt)
                schema_name = getattr(schema, '__name__', '') or ''
                chunk = _chunk_text_from_semantica_prompt(prompt) or getattr(_NER_CHUNK_TEXT, 'value', None) or ''
                _diag_record(
                    'llm_chunk_prompt',
                    schema=schema_name or 'unknown',
                    chunk_chars=len(chunk),
                    prompt_chars=len(augmented),
                    source_omitted_banners=(
                        augmented.count('repeated page banner omitted')
                        if 'TASK = ENTITIES' in (getattr(llm, '_semantica_domain_guidance', '') or '')
                        else 0
                    ),
                )
                result = original_typed(augmented, schema, **kwargs)
                if schema_name in {'EntitiesResponse', 'RelationsResponse', 'TripletsResponse'}:
                    store = getattr(llm, '_semantica_typed_store', None)
                    # Accumulate across chunks via _typed_set (do not overwrite).
                    _typed_set(schema_name, result, store if isinstance(store, dict) else None)
                    key = {
                        'EntitiesResponse': 'entities',
                        'RelationsResponse': 'relations',
                        'TripletsResponse': 'triplets',
                    }.get(schema_name, '')
                    n = len(getattr(result, key, None) or []) if key else 0
                    store_n = _typed_row_count(store.get(schema_name) if isinstance(store, dict) else result, key) if key else n
                    _diag_record(
                        'llm_typed_result',
                        schema=schema_name,
                        chunk_rows=n,
                        store_rows=store_n,
                    )
                return result

            llm.generate_typed = generate_typed  # type: ignore[method-assign]
        if hasattr(llm, 'generate_structured'):
            original_structured = llm.generate_structured

            def generate_structured(prompt: str, **kwargs):
                final_prompt = _augment(prompt) if _needs_augment(prompt) else prompt
                return original_structured(final_prompt, **kwargs)

            llm.generate_structured = generate_structured  # type: ignore[method-assign]
        llm._semantica_domain_guidance_hooked = True  # type: ignore[attr-defined]
    else:
        # Refresh per-call store even when generate_typed is already wrapped.
        llm._semantica_typed_store = call_store  # type: ignore[attr-defined]


def _with_guided_provider(
    domain_instructions: str,
    markers: tuple[str, ...],
    fn,
    *args,
    typed_store: dict[str, Any] | None = None,
    **kwargs,
):
    import semantica.semantic_extract.methods as methods

    guidance = (domain_instructions or '').strip()
    store = typed_store if typed_store is not None else {}
    _typed_store(store)
    if not guidance:
        return fn(*args, **kwargs)
    # Never put unfilled <<CHUNK_TEXT>> guidance into Bedrock `system` — Nova
    # otherwise extracts the placeholder token as a Document entity.
    if '<<CHUNK_TEXT>>' not in guidance and '<<TEXT>>' not in guidance:
        kwargs.setdefault('system', guidance[:2000])
    real_create = methods.create_provider

    def hooked_create(*cargs, **ckwargs):
        llm = real_create(*cargs, **ckwargs)
        _attach_domain_instructions(llm, guidance, markers=markers, typed_store=store)
        return llm

    methods.create_provider = hooked_create
    try:
        return fn(*args, **kwargs)
    finally:
        methods.create_provider = real_create


def register_bedrock_provider() -> None:
    """Register local BedrockProvider so Semantica create_provider('bedrock') works."""
    if provider_registry.get('bedrock') is not BedrockProvider:
        provider_registry.register('bedrock', BedrockProvider)


def install_guided_extract_hooks() -> None:
    """Wrap Semantica llm extractors so domain_instructions reach Bedrock."""
    global _GUIDED_HOOKS_INSTALLED
    if _GUIDED_HOOKS_INSTALLED:
        return
    register_bedrock_provider()
    import semantica.semantic_extract.methods as methods

    original_entities = methods.extract_entities_llm
    original_relations = methods.extract_relations_llm
    original_triplets = methods.extract_triplets_llm

    def extract_entities_llm(text: str, provider: str = 'openai', model=None, silent_fail: bool = False, max_text_length=None, structured_output_mode: str = 'typed', **kwargs):
        domain = kwargs.pop('domain_instructions', None) or ''
        source_file = kwargs.pop('source_file', None)
        typed_store: dict[str, Any] = {}
        prior_chunk = getattr(_NER_CHUNK_TEXT, 'value', None)
        _NER_CHUNK_TEXT.value = text
        try:
            entities = _with_guided_provider(
                domain, ('Extract named entities', 'entities'),
                original_entities, text,
                provider=provider, model=model, silent_fail=silent_fail,
                max_text_length=max_text_length, structured_output_mode=structured_output_mode,
                typed_store=typed_store, **kwargs,
            )
        finally:
            _NER_CHUNK_TEXT.value = prior_chunk
        typed = typed_store.get('EntitiesResponse') or _typed_pop('EntitiesResponse', typed_store)
        typed_n = _typed_row_count(typed, 'entities')
        print(f'[stage] llm_parsed entities={typed_n}')
        entities = list(entities or [])
        print(f'[stage] semantica_conversion entities={len(entities)}')
        _diag_record(
            'ner_typed_vs_converted',
            typed_entities=typed_n,
            converted_entities=len(entities),
            dropped_count=max(0, typed_n - len(entities)),
        )
        # Semantica rebuilds Entity.metadata and drops custom LLM fields; restore locally.
        # Keep LLM primary/subtype as predicted — do not force-relabel by pattern.
        # Predicate/subtype legality is enforced later via YAML gates + evidence.
        # Prefer full-document text for evidence verify when chunking (unique offsets).
        full_doc = getattr(_DOC_NER_CONTEXT, 'text', None) or text
        merged = _merge_entities_from_typed(entities, typed, full_doc, source_file=source_file)
        return _ground_entities_in_source(merged, full_doc, policy=load_policy())

    def extract_relations_llm(text: str, entities=None, provider: str = 'openai', model=None, silent_fail: bool = False, max_text_length=None, structured_output_mode: str = 'typed', max_retries: int = 3, extract_temporal_bounds: bool = False, **kwargs):
        domain = kwargs.pop('domain_instructions', None) or ''
        source_file = kwargs.pop('source_file', None)
        typed_store: dict[str, Any] = {}
        relations = _with_guided_provider(
            domain, ('Extract relations between entities', 'relations'),
            original_relations, text, entities=entities,
            provider=provider, model=model, silent_fail=silent_fail,
            max_text_length=max_text_length, structured_output_mode=structured_output_mode,
            max_retries=max_retries, extract_temporal_bounds=extract_temporal_bounds,
            typed_store=typed_store, **kwargs,
        )
        typed = typed_store.get('RelationsResponse') or _typed_pop('RelationsResponse', typed_store)
        full_doc = getattr(_DOC_NER_CONTEXT, 'text', None) or text
        return _merge_relations_from_typed(list(relations or []), typed, full_doc, source_file=source_file)

    def extract_triplets_llm(text: str, entities=None, relations=None, provider: str = 'openai', model=None, silent_fail: bool = False, max_text_length=None, structured_output_mode: str = 'typed', max_retries: int = 3, **kwargs):
        domain = kwargs.pop('domain_instructions', None) or ''
        source_file = kwargs.pop('source_file', None)
        typed_store: dict[str, Any] = {}
        triplets = _with_guided_provider(
            domain, ('Extract RDF triplets', 'triplets'),
            original_triplets, text, entities=entities, relations=relations,
            provider=provider, model=model, silent_fail=silent_fail,
            max_text_length=max_text_length, structured_output_mode=structured_output_mode,
            max_retries=max_retries, typed_store=typed_store, **kwargs,
        )
        typed = typed_store.get('TripletsResponse') or _typed_pop('TripletsResponse', typed_store)
        full_doc = getattr(_DOC_NER_CONTEXT, 'text', None) or text
        return _merge_triplets_from_typed(list(triplets or []), typed, full_doc, source_file=source_file)

    MethodRegistry.register('entity', 'llm', extract_entities_llm)
    MethodRegistry.register('relation', 'llm', extract_relations_llm)
    MethodRegistry.register('triplet', 'llm', extract_triplets_llm)
    _GUIDED_HOOKS_INSTALLED = True

# ---------------------------------------------------------------------------
# 10) Identity & dedupe — merge-safe entity helpers used by FactExtractor
# ---------------------------------------------------------------------------
@dataclass
class ExtractionBundle:
    entities: list[dict] = field(default_factory=list)
    raw_entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    triplets: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    identity_map: dict[str, str] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)
    # Confirmed evidence-grounded graph vs structured review quarantine.
    graph: dict[str, Any] = field(default_factory=dict)
    quarantine: dict[str, Any] = field(default_factory=dict)


_TECHNICAL_LABELS = frozenset({'system', 'equipment', 'component'})


def _meta_of(row: dict) -> dict:
    src = row.get('_src')
    if src is not None:
        meta = getattr(src, 'metadata', None)
        if isinstance(meta, dict):
            return meta
    meta = row.get('metadata')
    return meta if isinstance(meta, dict) else {}


def _identifier_values(meta: dict | None) -> set[str]:
    values: set[str] = set()
    for item in (meta or {}).get('identifiers') or []:
        if isinstance(item, dict):
            token = collapse(str(item.get('value') or ''))
        else:
            token = collapse(str(item or ''))
        if token:
            values.add(token.casefold())
    return values


def _alias_values(meta: dict | None) -> set[str]:
    values: set[str] = set()
    for item in (meta or {}).get('aliases') or []:
        token = collapse(str(item or ''))
        if token:
            values.add(token.casefold())
    return values


def _identity_linked(a: dict, b: dict) -> bool:
    """True only with exact surface, explicit alias, or shared identifier — not similarity alone."""
    text_a = collapse(str(a.get('text') or a.get('name') or '')).casefold()
    text_b = collapse(str(b.get('text') or b.get('name') or '')).casefold()
    if not text_a or not text_b:
        return False
    if text_a == text_b:
        return True
    meta_a, meta_b = _meta_of(a), _meta_of(b)
    if text_a in _alias_values(meta_b) or text_b in _alias_values(meta_a):
        return True
    ids_a, ids_b = _identifier_values(meta_a), _identifier_values(meta_b)
    return bool(ids_a and ids_b and not ids_a.isdisjoint(ids_b))


def _distinct_technical_identities(a: dict, b: dict) -> bool:
    """True when System/Equipment rows look like different objects (do not similarity-merge)."""
    la = collapse(str(a.get('label') or a.get('type') or '')).casefold()
    lb = collapse(str(b.get('label') or b.get('type') or '')).casefold()
    if la not in _TECHNICAL_LABELS and lb not in _TECHNICAL_LABELS:
        return False
    text_a = collapse(str(a.get('text') or a.get('name') or '')).casefold()
    text_b = collapse(str(b.get('text') or b.get('name') or '')).casefold()
    if not text_a or not text_b or text_a == text_b:
        return False
    if _identity_linked(a, b):
        return False
    ids_a, ids_b = _identifier_values(_meta_of(a)), _identifier_values(_meta_of(b))
    if ids_a and ids_b and ids_a.isdisjoint(ids_b):
        return True
    # Both technical labels, different surfaces, no shared identity/alias → refuse name-only merge.
    return la in _TECHNICAL_LABELS and lb in _TECHNICAL_LABELS


def _group_safe_to_merge(members: list[dict]) -> bool:
    """Merge only when every pair has identity evidence (never similarity-only)."""
    for index, left in enumerate(members or []):
        for right in members[index + 1 :]:
            if _distinct_technical_identities(left, right):
                return False
            if not _identity_linked(left, right):
                return False
    return True


def _identity_resolution_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load identity_resolution flags + type_equivalence_groups (YAML-only)."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    cfg = data.get('identity_resolution') if isinstance(data.get('identity_resolution'), dict) else {}
    groups: list[set[str]] = []
    for row in cfg.get('type_equivalence_groups') or []:
        if not isinstance(row, (list, tuple)):
            continue
        bucket = {collapse(str(x or '')).casefold() for x in row if collapse(str(x or ''))}
        if bucket:
            groups.append(bucket)
    roles = cfg.get('identifier_roles') if isinstance(cfg.get('identifier_roles'), dict) else {}
    identity_merge_roles: set[str] = set()
    for role, spec in roles.items():
        if not isinstance(spec, dict):
            continue
        if bool(spec.get('identity_merge')):
            identity_merge_roles.add(collapse(str(role or '')).casefold())
    if not identity_merge_roles:
        identity_merge_roles.add('identity')
    patterns = []
    for row in cfg.get('identifier_patterns') or []:
        if isinstance(row, dict) and str(row.get('pattern') or '').strip():
            patterns.append(row)
    return {
        'require_type_compatibility': bool(cfg.get('require_type_compatibility', True)),
        'merge_identity_codes_despite_type_conflict': bool(
            cfg.get('merge_identity_codes_despite_type_conflict', False)
        ),
        'require_alias_source_evidence': bool(cfg.get('require_alias_source_evidence', True)),
        'prefer_typed_survivor_on_identity_link': bool(
            cfg.get('prefer_typed_survivor_on_identity_link', True)
        ),
        'ambiguous_to_review': bool(cfg.get('ambiguous_to_review', True)),
        'merge_on_fuzzy_similarity_alone': bool(cfg.get('merge_on_fuzzy_similarity_alone', False)),
        'type_equivalence_groups': groups,
        'identity_merge_roles': identity_merge_roles,
        'identifier_patterns': patterns,
    }


def _entity_primary_label(entity: dict) -> str:
    return collapse(str(entity.get('label') or entity.get('type') or ''))


def _labels_type_compatible(label_a: str, label_b: str, settings: dict[str, Any]) -> bool:
    a = collapse(label_a or '').casefold()
    b = collapse(label_b or '').casefold()
    if not a or not b:
        return True
    if a == b:
        return True
    for group in settings.get('type_equivalence_groups') or []:
        if a in group and b in group:
            return True
    return False


def _identity_role_identifier_values(entity: dict, settings: dict[str, Any]) -> set[str]:
    """Explicit identifier tokens whose YAML role allows identity_merge (conflict keys)."""
    keys: set[str] = set()
    meta = _meta_of(entity)
    roles_ok = settings.get('identity_merge_roles') or {'identity'}
    for item in meta.get('identifiers') or []:
        if isinstance(item, dict):
            token = collapse(str(item.get('value') or ''))
            role = collapse(str(item.get('role') or 'identity')).casefold() or 'identity'
        else:
            token = collapse(str(item or ''))
            role = 'identity'
        if token and role in roles_ok:
            keys.add(token.casefold())
    return keys


def _identity_merge_keys(entity: dict, *, policy: Policy | None, settings: dict[str, Any]) -> set[str]:
    """Source identity keys eligible for merge (YAML identity_merge roles / patterns)."""
    keys = set(_identity_role_identifier_values(entity, settings))
    patterns = settings.get('identifier_patterns') or []
    pattern_rows = [{'pattern': r.get('pattern'), 'kind': r.get('kind')} for r in patterns]
    for token in list(keys):
        if _match_identifier_kind(token, pattern_rows):
            keys.add(token)
    # Full-string identity tags claimed by this entity (policy patterns).
    for tag in _entity_identity_tags(entity, policy=policy):
        if tag:
            keys.add(tag.casefold())
    return keys


def _alias_has_source_evidence(
    entity: dict,
    alias: str,
    *,
    document_text: str = '',
    settings: dict[str, Any] | None = None,
) -> bool:
    """Exact alias must appear in entity evidence or document when YAML requires it."""
    cfg = settings or {}
    if not cfg.get('require_alias_source_evidence', True):
        return True
    needle = collapse(alias or '')
    if not needle:
        return False
    meta = _meta_of(entity)
    evidence = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else {}
    quote = str(evidence.get('source_text') or '')
    if _object_verbatim_in_evidence(needle, quote):
        return True
    if document_text and _object_verbatim_in_evidence(needle, document_text):
        return True
    # Peer provenance quotes.
    for ref in meta.get('source_references') or []:
        if not isinstance(ref, dict):
            continue
        if _object_verbatim_in_evidence(needle, str(ref.get('source_text') or '')):
            return True
    return False


def _entities_safe_identity_merge(
    left: dict,
    right: dict,
    *,
    policy: Policy | None,
    settings: dict[str, Any],
    document_text: str = '',
) -> bool:
    """Merge only with unambiguous identity proof — never fuzzy/name-only.

    Exact aliases may candidate-pair, but still need source-backed proof + type
    compatibility. Conflicting unique identity keys always block.
    """
    # Conflicting unique identifiers (YAML identity_merge roles) always block,
    # even when aliases match. Do not use full-surface identity tags here — those
    # are positive merge keys only (avoids false conflicts on titled duplicates).
    conflict_a = _identity_role_identifier_values(left, settings)
    conflict_b = _identity_role_identifier_values(right, settings)
    if conflict_a and conflict_b and conflict_a.isdisjoint(conflict_b):
        return False

    keys_a = _identity_merge_keys(left, policy=policy, settings=settings)
    keys_b = _identity_merge_keys(right, policy=policy, settings=settings)
    shared = bool(keys_a and keys_b and not keys_a.isdisjoint(keys_b))
    text_a = collapse(str(left.get('text') or left.get('name') or '')).casefold()
    text_b = collapse(str(right.get('text') or right.get('name') or '')).casefold()
    meta_a, meta_b = _meta_of(left), _meta_of(right)
    alias_hit = False
    if text_a and text_a in _alias_values(meta_b):
        alias_hit = _alias_has_source_evidence(
            right, text_a, document_text=document_text, settings=settings,
        )
    if text_b and text_b in _alias_values(meta_a):
        alias_hit = alias_hit or _alias_has_source_evidence(
            left, text_b, document_text=document_text, settings=settings,
        )
    # Same surface is identity proof (exact, not fuzzy).
    same_surface = bool(text_a and text_a == text_b)
    # Never merge on fuzzy / similar names alone.
    if not (shared or alias_hit or same_surface):
        return False
    if settings.get('require_type_compatibility', True):
        if not _labels_type_compatible(
            _entity_primary_label(left), _entity_primary_label(right), settings,
        ):
            if not settings.get('merge_identity_codes_despite_type_conflict', False):
                return False
    return True


def _pick_identity_merge_survivor(members: list[dict], settings: dict[str, Any]) -> dict:
    """Prefer typed / higher-confidence / shorter canonical surface (YAML-guided)."""
    prefer_typed = bool(settings.get('prefer_typed_survivor_on_identity_link', True))

    def _score(ent: dict) -> tuple:
        label = _entity_primary_label(ent)
        typed = 1 if label and label.casefold() not in {'other', 'entity', 'concept'} else 0
        conf = float(ent.get('confidence') or 0)
        text = collapse(str(ent.get('text') or ''))
        # Prefer compact identity surfaces (tag) over long titles when both typed.
        return (typed if prefer_typed else 0, conf, -len(text), text.casefold())

    return max(members, key=_score)


def _apply_dict_entity_merge(survivor: dict, members: list[dict]) -> None:
    """Fold aliases/identifiers/evidence/provenance into survivor dict entity."""
    meta = dict(survivor.get('metadata') if isinstance(survivor.get('metadata'), dict) else {})
    canonical_id = collapse(str(survivor.get('id') or meta.get('canonical_id') or ''))
    canonical_text = collapse(str(survivor.get('text') or ''))
    aliases = list(meta.get('aliases') or [])
    identifiers = list(meta.get('identifiers') or [])
    properties = meta.get('properties')
    source_refs = list(meta.get('source_references') or []) if isinstance(meta.get('source_references'), list) else []
    primary = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else None
    primary_quote = collapse(str((primary or {}).get('source_text') or ''))
    merged_from: list[str] = list(meta.get('merged_from_ids') or [])

    for member in members:
        mid = collapse(str(member.get('id') or ''))
        if mid and mid != canonical_id and mid not in merged_from:
            merged_from.append(mid)
        member_meta = _meta_of(member)
        aliases.extend(list(member_meta.get('aliases') or []))
        surface = collapse(str(member.get('text') or member.get('name') or ''))
        if surface and surface.casefold() != canonical_text.casefold():
            aliases.append(surface)
        identifiers.extend(list(member_meta.get('identifiers') or []))
        properties = _merge_properties(properties, member_meta.get('properties'))
        peer_evidence = member_meta.get('evidence')
        peer_quote = (
            collapse(str((peer_evidence or {}).get('source_text') or ''))
            if isinstance(peer_evidence, dict) else ''
        )
        if peer_quote and peer_quote != primary_quote:
            _append_source_reference(
                source_refs,
                peer_evidence,
                from_entity_id=mid,
                section_context=member_meta.get('section_context'),
                provenance=member_meta.get('provenance'),
            )
        elif isinstance(member_meta.get('section_context'), dict) and not meta.get('section_context'):
            meta['section_context'] = dict(member_meta['section_context'])

    meta['aliases'] = _dedupe_aliases(aliases, canonical_text=canonical_text)
    meta['identifiers'] = _dedupe_identifiers(identifiers)
    if isinstance(properties, dict) and properties:
        meta['properties'] = properties
    meta['source_references'] = source_refs
    meta['canonical_id'] = canonical_id
    meta['merged_from_ids'] = merged_from
    provenance = meta.get('provenance') if isinstance(meta.get('provenance'), dict) else {}
    provenance = dict(provenance)
    provenance['identity_merge'] = True
    provenance['merged_member_count'] = len(merged_from) + 1
    meta['provenance'] = provenance
    survivor['metadata'] = meta
    survivor['id'] = canonical_id


def merge_identity_duplicate_entities(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    document_text: str = '',
    source_file: str | None = None,
) -> dict[str, str]:
    """Merge unambiguous identity duplicates using YAML identity_resolution rules.

    Never merges on fuzzy/name similarity alone. Conflicting identity keys block
    merges even when aliases match. Returns old_id → canonical_id map.
    """
    del source_file  # reserved for provenance/diagnostics
    settings = _identity_resolution_settings(policy)
    entities = [e for e in (bundle.entities or []) if isinstance(e, dict)]
    if len(entities) < 2:
        return {}

    # Union-find over pairwise-safe merges.
    parent: dict[int, int] = {i: i for i in range(len(entities))}

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[rj] = ri

    for i, left in enumerate(entities):
        for j in range(i + 1, len(entities)):
            right = entities[j]
            if _entities_safe_identity_merge(
                left, right, policy=policy, settings=settings, document_text=document_text,
            ):
                _union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(entities)):
        clusters.setdefault(_find(i), []).append(i)

    id_map: dict[str, str] = {}
    kept: list[dict] = []
    ambiguous = 0
    for members_idx in clusters.values():
        members = [entities[i] for i in members_idx]
        if len(members) == 1:
            kept.append(members[0])
            continue
        # Ambiguous: more than one pairwise-unlinked subgroup should not happen
        # after union-find; still require every pair safe (defensive).
        if not all(
            _entities_safe_identity_merge(
                members[a], members[b], policy=policy, settings=settings, document_text=document_text,
            )
            for a in range(len(members))
            for b in range(a + 1, len(members))
        ):
            ambiguous += 1
            if settings.get('ambiguous_to_review', True):
                bundle.warnings.append(
                    'identity merge skipped (ambiguous / conflicting identities): '
                    + ', '.join(repr(collapse(str(m.get('text') or ''))) for m in members[:4])
                )
            kept.extend(members)
            continue
        survivor = _pick_identity_merge_survivor(members, settings)
        _apply_dict_entity_merge(survivor, members)
        canon = collapse(str(survivor.get('id') or ''))
        for member in members:
            mid = collapse(str(member.get('id') or ''))
            if mid and canon:
                id_map[mid] = canon
        kept.append(survivor)

    bundle.entities = kept
    if id_map:
        bundle.identity_map.update(id_map)
        _diag_record(
            'identity_duplicate_merge',
            merged_ids=len(id_map),
            entities_kept=len(kept),
            ambiguous_clusters=ambiguous,
        )
    return id_map


def _redirect_bundle_endpoints_after_identity_merge(
    bundle: ExtractionBundle,
    id_map: dict[str, str],
) -> None:
    """Point relation/triplet endpoint ids at canonical survivors (surfaces unchanged)."""
    if not id_map:
        return
    by_id = {
        collapse(str(e.get('id') or '')): e
        for e in (bundle.entities or [])
        if isinstance(e, dict) and collapse(str(e.get('id') or ''))
    }

    def _rewrite(row: dict) -> None:
        for key in ('subject_id', 'object_id'):
            raw = collapse(str(row.get(key) or ''))
            if not raw:
                continue
            canon = id_map.get(raw, raw)
            row[key] = canon
        # If ids resolve, optionally sync endpoint text to canonical surface when
        # the old id was merged away (keeps Known Entity surfaces stable).
        for text_key, id_key in (('subject', 'subject_id'), ('object', 'object_id')):
            eid = collapse(str(row.get(id_key) or ''))
            ent = by_id.get(eid)
            if ent is None:
                continue
            # Do not rewrite attribute literals.
            if text_key == 'object' and collapse(str(row.get('object_value_kind') or '')).casefold() == 'literal':
                continue
            if collapse(str(row.get('relation_category') or '')).casefold() == 'attribute':
                if text_key == 'object':
                    continue

    # Endpoints are assigned later in validate/build; keep helper for post-validate remap.
    for pool in (bundle.relations, bundle.triplets):
        for row in pool or []:
            if isinstance(row, dict):
                _rewrite(row)


def _dedupe_identifiers(items: list) -> list:
    out: list = []
    seen: set[str] = set()
    for item in items or []:
        if isinstance(item, dict):
            token = collapse(str(item.get('value') or ''))
            key = token.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        else:
            token = collapse(str(item or ''))
            key = token.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(token)
    return out


def _dedupe_aliases(items: list, *, canonical_text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    canon = collapse(canonical_text or '').casefold()
    if canon:
        seen.add(canon)
    for item in items or []:
        token = collapse(str(item or ''))
        key = token.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def _merge_properties(base: Any, incoming: Any) -> dict:
    out: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    if isinstance(incoming, dict):
        for key, value in incoming.items():
            if key not in out or out[key] in (None, '', {}, []):
                out[key] = value
    elif isinstance(incoming, list):
        # Preserve list-shaped properties under a stable bag when needed.
        existing = out.get('_items')
        bag = list(existing) if isinstance(existing, list) else []
        for item in incoming:
            if item not in bag:
                bag.append(item)
        if bag:
            out['_items'] = bag
    return out


def _append_source_reference(
    refs: list[dict],
    evidence: Any,
    *,
    from_entity_id: str,
    section_context: Any = None,
    provenance: Any = None,
) -> None:
    if not isinstance(evidence, dict):
        return
    quote = evidence.get('source_text')
    if not isinstance(quote, str) or not quote:
        return
    ref = {
        'source_text': quote,
        'source_file': evidence.get('source_file'),
        'start_char': evidence.get('start_char'),
        'end_char': evidence.get('end_char'),
        'from_entity_id': from_entity_id,
    }
    if isinstance(section_context, dict) and section_context:
        ref['section_context'] = dict(section_context)
    if isinstance(provenance, dict) and provenance:
        ref['provenance'] = dict(provenance)
    signature = (
        collapse(quote),
        str(ref.get('source_file') or ''),
        str(ref.get('start_char')),
        str(ref.get('end_char')),
        from_entity_id,
    )
    existing = {
        (
            collapse(str(item.get('source_text') or '')),
            str(item.get('source_file') or ''),
            str(item.get('start_char')),
            str(item.get('end_char')),
            str(item.get('from_entity_id') or ''),
        )
        for item in refs
        if isinstance(item, dict)
    }
    if signature not in existing:
        refs.append(ref)


def _apply_merge_to_representative(
    representative,
    *,
    members: list[dict],
    merged_entity: dict,
    canonical_id: str,
) -> None:
    """Keep original Entity object; fold verified metadata from the merge group into it."""
    meta = dict(getattr(representative, 'metadata', None) or {})
    canonical_text = collapse(getattr(representative, 'text', '') or '')
    aliases = list(meta.get('aliases') or [])
    identifiers = list(meta.get('identifiers') or [])
    properties = meta.get('properties')
    source_refs = list(meta.get('source_references') or []) if isinstance(meta.get('source_references'), list) else []

    # Representative evidence stays put; peer evidence becomes source_references.
    primary = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else None
    primary_quote = collapse(str((primary or {}).get('source_text') or ''))

    for member in members:
        member_id = str(member.get('id') or '')
        member_meta = _meta_of(member)
        aliases.extend(list(member_meta.get('aliases') or []))
        surface = collapse(str(member.get('text') or member.get('name') or ''))
        if surface and surface.casefold() != canonical_text.casefold():
            aliases.append(surface)
        identifiers.extend(list(member_meta.get('identifiers') or []))
        properties = _merge_properties(properties, member_meta.get('properties'))
        peer_evidence = member_meta.get('evidence')
        peer_quote = collapse(str((peer_evidence or {}).get('source_text') or '')) if isinstance(peer_evidence, dict) else ''
        if peer_quote and peer_quote != primary_quote:
            _append_source_reference(
                source_refs,
                peer_evidence,
                from_entity_id=member_id,
                section_context=member_meta.get('section_context'),
                provenance=member_meta.get('provenance'),
            )
        elif isinstance(member_meta.get('section_context'), dict) and not meta.get('section_context'):
            meta['section_context'] = dict(member_meta['section_context'])

    merged_meta = merged_entity.get('metadata') if isinstance(merged_entity.get('metadata'), dict) else {}
    aliases.extend(list(merged_meta.get('aliases') or []))
    identifiers.extend(list(merged_meta.get('identifiers') or []))
    properties = _merge_properties(properties, merged_meta.get('properties'))
    if isinstance(merged_meta.get('provenance'), dict):
        existing_prov = meta.get('provenance') if isinstance(meta.get('provenance'), dict) else {}
        meta['provenance'] = {**dict(merged_meta['provenance']), **dict(existing_prov)}
    if isinstance(merged_meta.get('section_context'), dict) and not meta.get('section_context'):
        meta['section_context'] = dict(merged_meta['section_context'])

    meta['aliases'] = _dedupe_aliases(aliases, canonical_text=canonical_text)
    meta['identifiers'] = _dedupe_identifiers(identifiers)
    if isinstance(properties, dict) and properties:
        meta['properties'] = properties
    meta['source_references'] = source_refs
    meta['canonical_id'] = canonical_id
    meta['merged_from_ids'] = [str(m.get('id') or '') for m in members if m.get('id')]
    representative.metadata = meta


def _entity_identity_tags(
    entity: dict,
    *,
    policy: Policy | None = None,
) -> set[str]:
    """Engineering tags this entity uniquely claims (identifiers + single-tag surfaces).

    Uses the full identifier/surface string as the identity key. Never claims a
    shorter regex capture alone (so ``22-K-3901A-C`` does not also own ``22-K-3901A``).
    """
    tags: set[str] = set()
    meta = entity.get('metadata') if isinstance(entity.get('metadata'), dict) else {}

    def _claim_identity_surface(raw: str) -> None:
        text = collapse(raw or '')
        if not text:
            return
        found = _engineering_tags_in(text, policy=policy)
        if not found:
            return
        # Exact full-string identity only — reject prefix captures.
        tags.add(_normalize_equipment_tag(text))

    for ident in _identifier_values(meta):
        _claim_identity_surface(ident)
    surface = collapse(str(entity.get('text') or entity.get('name') or ''))
    _claim_identity_surface(surface)
    return tags


def build_endpoint_resolution_index(
    entities: list[dict],
    *,
    policy: Policy | None = None,
) -> dict[str, Any]:
    """Exact surface/alias/id index plus unique engineering-tag → entity map.

    Unique-tag resolution supports endpoints like ``22-PK-3904: Title`` when a
    single canonical entity claims that tag. Multi-tag compound titles do not
    claim uniqueness for every embedded tag (avoids unsafe merges).
    """
    by_id: dict[str, dict] = {}
    by_surface: dict[str, dict] = {}
    tag_owners: dict[str, list[str]] = {}
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        eid = collapse(str(entity.get('id') or entity.get('metadata', {}).get('canonical_id') or ''))
        if not eid:
            meta = entity.get('metadata') if isinstance(entity.get('metadata'), dict) else {}
            eid = collapse(str(meta.get('canonical_id') or ''))
        surface = collapse(str(entity.get('text') or entity.get('name') or ''))
        if eid:
            by_id[eid] = entity
        if surface:
            by_surface[surface.casefold()] = entity
            meta = entity.get('metadata') if isinstance(entity.get('metadata'), dict) else {}
            for alias in meta.get('aliases') or []:
                alias_key = collapse(str(alias or '')).casefold()
                if alias_key and alias_key not in by_surface:
                    by_surface[alias_key] = entity
            for ident in _identifier_values(meta):
                if ident and ident not in by_surface:
                    by_surface[ident] = entity
        if eid:
            claimed = _entity_identity_tags(entity, policy=policy)
            # Short tags = identity keys that are themselves a single engineering tag.
            short_tags: set[str] = set()
            for tag in claimed:
                found = _engineering_tags_in(tag, policy=policy)
                if (
                    len(found) == 1
                    and _normalize_equipment_tag(found[0]) == tag
                ):
                    short_tags.add(tag)
            # Multi-tag compounds must not claim uniqueness for each member tag.
            register = claimed - short_tags if len(short_tags) > 1 else claimed
            for tag in register:
                if tag:
                    tag_owners.setdefault(tag, []).append(eid)
    by_unique_tag: dict[str, dict] = {}
    for tag, owners in tag_owners.items():
        unique = list(dict.fromkeys(owners))
        if len(unique) == 1 and unique[0] in by_id:
            by_unique_tag[tag] = by_id[unique[0]]
    return {
        'by_id': by_id,
        'by_surface': by_surface,
        'by_unique_tag': by_unique_tag,
        'policy': policy,
    }


def _entity_lookup(entities: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    index = build_endpoint_resolution_index(entities)
    return index['by_id'], index['by_surface']


def _resolve_endpoint_ref(
    value: str,
    *,
    by_surface: dict[str, dict],
    attribute_mode: bool = False,
    index: dict[str, Any] | None = None,
    policy: Policy | None = None,
) -> tuple[str | None, str]:
    """Return (entity_id_or_none, kind) where kind is entity|literal|unresolved.

    Exact surface / alias / identifier / unique full-tag only.
    Never prefix-merge (e.g. ``22-K-3901A`` must not resolve to ``22-K-3901A-C``).
    """
    key = collapse(value or '').casefold()
    if not key:
        return None, 'unresolved'
    hit = by_surface.get(key)
    if hit is not None:
        return collapse(str(hit.get('id') or (hit.get('metadata') or {}).get('canonical_id') or '')) or None, 'entity'
    idx = index or {}
    by_unique = idx.get('by_unique_tag') if isinstance(idx.get('by_unique_tag'), dict) else {}
    pol = policy if policy is not None else idx.get('policy')
    # Exact unique-tag hit (full endpoint string, or unique tag embedded in
    # ``TAG: Title``). Entities claim full identifier surfaces only, so a shorter
    # endpoint like ``22-K-3901A`` cannot resolve to owner of ``22-K-3901A-C``.
    full = _normalize_equipment_tag(value)
    if full and full in by_unique:
        ent = by_unique[full]
        eid = collapse(str(ent.get('id') or (ent.get('metadata') or {}).get('canonical_id') or ''))
        if eid:
            return eid, 'entity'
    tags = [_normalize_equipment_tag(t) for t in _engineering_tags_in(value, policy=pol)]
    tags = [t for t in tags if t]
    if len(tags) == 1 and tags[0] in by_unique:
        ent = by_unique[tags[0]]
        eid = collapse(str(ent.get('id') or (ent.get('metadata') or {}).get('canonical_id') or ''))
        if eid:
            return eid, 'entity'
    if attribute_mode:
        return None, 'literal'
    return None, 'unresolved'


def _canonicalize_endpoint(
    value: str,
    surface_map: dict[str, str],
    *,
    index: dict[str, Any] | None = None,
    policy: Policy | None = None,
) -> str:
    """Map endpoint mention → canonical entity surface when uniquely resolvable."""
    key = collapse(value or '').casefold()
    if not key:
        return ''
    if key in surface_map:
        return surface_map[key]
    if index:
        eid, kind = _resolve_endpoint_ref(
            value,
            by_surface=index.get('by_surface') or {},
            index=index,
            policy=policy,
        )
        if kind == 'entity' and eid:
            ent = (index.get('by_id') or {}).get(eid) or {}
            canon = collapse(str(ent.get('text') or ent.get('name') or ''))
            if canon:
                return canon
    return collapse(value)


def _build_surface_map_from_entities(
    entities: list[dict],
    *,
    policy: Policy | None = None,
) -> dict[str, str]:
    """surface/alias/identifier/unique-tag → canonical entity text."""
    surface_map: dict[str, str] = {}
    index = build_endpoint_resolution_index(entities, policy=policy)
    for entity in entities or []:
        canon = collapse(str(entity.get('text') or entity.get('name') or ''))
        if not canon:
            continue
        surface_map[canon.casefold()] = canon
        meta = entity.get('metadata') if isinstance(entity.get('metadata'), dict) else {}
        for alias in _alias_values(meta):
            surface_map.setdefault(alias, canon)
        for ident in _identifier_values(meta):
            surface_map.setdefault(ident, canon)
    for tag, ent in (index.get('by_unique_tag') or {}).items():
        canon = collapse(str(ent.get('text') or ent.get('name') or ''))
        if tag and canon:
            surface_map.setdefault(tag, canon)
    return surface_map


def hierarchy_parent_as_subject_predicates(policy: Policy) -> set[str]:
    return {
        p.casefold()
        for p in _policy_labels(policy, 'hierarchy', 'parent_as_subject_markers')
    }


def hierarchy_child_as_subject_predicates(policy: Policy) -> set[str]:
    """Inverse hierarchy markers (child→parent wording) from YAML markers."""
    parentish = hierarchy_parent_as_subject_predicates(policy)
    all_hier = hierarchy_predicate_set(policy)
    # Markers that typically read child-as-subject when not in parent-as-subject list.
    inverse_like = {
        'part of', 'partof', 'belongs to', 'belongsto', 'contained in', 'containedin',
        'member of', 'memberof', 'child of', 'childof', 'property of', 'propertyof',
    }
    # Include YAML hierarchy markers that match inverse phrasing, plus compact
    # camelCase aliases (partOf) even when only spaced forms appear in YAML.
    out = {p for p in all_hier if p in inverse_like and p not in parentish}
    out |= {p for p in inverse_like if p not in parentish}
    # Also honor explicit attribute inverse list when present.
    out |= {
        p.casefold()
        for p in _policy_labels(policy, 'hierarchy', 'attribute_inverse_predicate_markers')
    }
    return out


def canonicalize_hierarchy_edge(
    subject_id: str,
    object_id: str,
    predicate: str,
    *,
    policy: Policy,
) -> tuple[str, str, str]:
    """Return (parent_id, child_id, orientation) without inventing links.

    orientation is ``as_extracted`` or ``inverted_for_hierarchy``.
    """
    pred = collapse(predicate or '').casefold()
    if not subject_id or not object_id or not pred:
        return subject_id, object_id, 'as_extracted'
    if not bool((policy.data.get('hierarchy') or {}).get('canonicalize_hierarchy_direction', True)):
        return subject_id, object_id, 'as_extracted'
    if pred in hierarchy_child_as_subject_predicates(policy):
        return object_id, subject_id, 'inverted_for_hierarchy'
    if pred in hierarchy_parent_as_subject_predicates(policy):
        return subject_id, object_id, 'as_extracted'
    # Default: treat hierarchy markers as parent→child as extracted.
    if pred in hierarchy_predicate_set(policy):
        return subject_id, object_id, 'as_extracted'
    return subject_id, object_id, 'as_extracted'


def classify_relation_category(predicate: str, *, policy: Policy) -> str:
    """Bucket predicates so physical hierarchy rules are not over-applied."""
    pred = collapse(predicate or '').casefold()
    if not pred:
        return 'unknown'
    if pred in {'documentcontains', 'sectioncontains', 'has document', 'hasdocument'}:
        return 'document_hierarchy'
    if pred in hierarchy_predicate_set(policy):
        return 'engineering_hierarchy'
    technical = {
        p.casefold()
        for p in _policy_labels(policy, 'hierarchy', 'technical_predicate_markers')
    }
    if pred in technical:
        return 'technical'
    location = {
        p.casefold()
        for p in _policy_labels(policy, 'hierarchy', 'location_predicate_markers')
    }
    if pred in location:
        return 'location'
    if pred in {p.casefold() for p in attribute_predicate_hints(policy)}:
        return 'attribute'
    return 'semantic'


def _claim_evidence_recovery_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load claim_evidence_recovery flags (nearby span + membership/section/table gates)."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    cfg = data.get('claim_evidence_recovery') if isinstance(data.get('claim_evidence_recovery'), dict) else {}
    try:
        max_chars = int(cfg.get('nearby_span_max_chars', 900))
    except (TypeError, ValueError):
        max_chars = 900
    hierarchy = data.get('hierarchy') if isinstance(data.get('hierarchy'), dict) else {}
    try:
        min_pipe = int(hierarchy.get('flat_list_min_pipe_cells', 2))
    except (TypeError, ValueError):
        min_pipe = 2
    action_cues: list[str] = []
    for item in cfg.get('incomplete_attribute_action_cues') or []:
        cue = collapse(str(item or '')).casefold()
        if cue and cue not in action_cues:
            action_cues.append(cue)
    return {
        'nearby_span_max_chars': max(80, min(max_chars, 4000)),
        'require_membership_evidence': bool(cfg.get('require_membership_evidence', True)),
        'require_section_ownership': bool(cfg.get('require_section_ownership', True)),
        'require_table_context': bool(cfg.get('require_table_context', True)),
        'require_table_context_for_identifier_objects': bool(
            cfg.get('require_table_context_for_identifier_objects', True)
        ),
        'reuse_section_membership_markers': bool(cfg.get('reuse_section_membership_markers', True)),
        'require_predicate_meaning': bool(cfg.get('require_predicate_meaning', True)),
        'require_claim_direction': bool(cfg.get('require_claim_direction', True)),
        'reject_incomplete_attribute_action_phrases': bool(
            cfg.get('reject_incomplete_attribute_action_phrases', True)
        ),
        'incomplete_attribute_action_cues': action_cues,
        # Creation-time quotes required; Python verifies absolute offsets (no invent).
        'require_creation_evidence': bool(cfg.get('require_creation_evidence', True)),
        'min_pipe_cells': max(2, min_pipe),
    }


def _claim_nearby_span_max_chars(policy: Policy | None = None) -> int:
    """Max contiguous chars for nearby endpoint-span recovery (policy-tunable)."""
    return int(_claim_evidence_recovery_settings(policy)['nearby_span_max_chars'])


def _semantic_relation_recovery_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load semantic_relation_recovery knobs (indexes, sync, caps)."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    cfg = data.get('semantic_relation_recovery') if isinstance(
        data.get('semantic_relation_recovery'), dict
    ) else {}
    idx = cfg.get('evidence_index') if isinstance(cfg.get('evidence_index'), dict) else {}
    windows = [
        collapse(str(w or '')).casefold()
        for w in (idx.get('windows') or ['sentence', 'table_row', 'section_block'])
        if collapse(str(w or ''))
    ]
    try:
        max_repairs = int(cfg.get('max_repairs_per_doc', 200))
    except (TypeError, ValueError):
        max_repairs = 200
    try:
        max_candidates = int(cfg.get('max_candidates_per_doc', 80))
    except (TypeError, ValueError):
        max_candidates = 80
    try:
        table_max = int(idx.get('table_row_max_chars', 800))
    except (TypeError, ValueError):
        table_max = 800
    try:
        section_max = int(idx.get('section_block_max_chars', 1200))
    except (TypeError, ValueError):
        section_max = 1200
    try:
        sentence_max = int(idx.get('sentence_max_chars', 600))
    except (TypeError, ValueError):
        sentence_max = 600
    try:
        cand_max = int(cfg.get('candidate_max_window_chars', 420))
    except (TypeError, ValueError):
        cand_max = 420
    try:
        cue_ctx = int(cfg.get('candidate_cue_context_chars', 180))
    except (TypeError, ValueError):
        cue_ctx = 180
    return {
        'enabled': bool(cfg.get('enabled', True)),
        'sync_quarantine_after_recovery': bool(cfg.get('sync_quarantine_after_recovery', True)),
        'max_repairs_per_doc': max(0, min(max_repairs, 2000)),
        'max_candidates_per_doc': max(0, min(max_candidates, 1000)),
        'prefer_local_windows': bool(cfg.get('prefer_local_windows', True)),
        'candidate_max_window_chars': max(80, min(cand_max, 2000)),
        'candidate_cue_context_chars': max(40, min(cue_ctx, 1000)),
        'evidence_index_enabled': bool(idx.get('enabled', True)),
        'windows': windows or ['sentence', 'table_row', 'section_block'],
        'table_row_max_chars': max(80, min(table_max, 4000)),
        'section_block_max_chars': max(120, min(section_max, 8000)),
        'sentence_max_chars': max(40, min(sentence_max, 2000)),
    }


def _predicate_alias_map(policy: Policy | None = None) -> dict[str, str]:
    """Flatten hierarchy.predicate_aliases + contract aliases → canonical predicate."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    hierarchy = data.get('hierarchy') if isinstance(data.get('hierarchy'), dict) else {}
    aliases = hierarchy.get('predicate_aliases') if isinstance(
        hierarchy.get('predicate_aliases'), dict
    ) else {}
    out: dict[str, str] = {}
    for raw, canon in aliases.items():
        key = collapse(str(raw or '')).casefold()
        val = collapse(str(canon or ''))
        if key and val:
            out[key] = val
    contracts = data.get('predicate_contracts') if isinstance(
        data.get('predicate_contracts'), dict
    ) else {}
    for canon_key, contract in contracts.items():
        canon = collapse(str(canon_key or ''))
        if not canon:
            continue
        out.setdefault(canon.casefold(), canon)
        if not isinstance(contract, dict):
            continue
        for alias in contract.get('aliases') or []:
            key = collapse(str(alias or '')).casefold()
            if key:
                out[key] = canon
    return out


def canonical_predicate(predicate: str, *, policy: Policy | None = None) -> str:
    """Normalize predicate surface via YAML aliases/contracts (no hardcoded maps)."""
    pred = collapse(predicate or '')
    if not pred:
        return ''
    aliases = _predicate_alias_map(policy)
    return aliases.get(pred.casefold(), pred)


def predicate_contract(predicate: str, *, policy: Policy | None = None) -> dict[str, Any]:
    """Return YAML predicate_contracts entry for the canonical predicate, else {}."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    contracts = data.get('predicate_contracts') if isinstance(
        data.get('predicate_contracts'), dict
    ) else {}
    if not contracts:
        return {}
    canon = canonical_predicate(predicate, policy=pol)
    if not canon:
        return {}
    direct = contracts.get(canon)
    if isinstance(direct, dict):
        out = dict(direct)
        out['_canonical'] = canon
        return out
    # Case-insensitive lookup.
    key = canon.casefold()
    for raw_key, contract in contracts.items():
        if collapse(str(raw_key or '')).casefold() == key and isinstance(contract, dict):
            out = dict(contract)
            out['_canonical'] = collapse(str(raw_key))
            return out
    return {}


def _contract_endpoint_primaries(contract: dict[str, Any], role: str) -> set[str]:
    endpoints = contract.get('endpoints') if isinstance(contract.get('endpoints'), dict) else {}
    key = 'subject_primaries' if role == 'subject' else 'object_primaries'
    return {
        collapse(str(p or '')).casefold()
        for p in (endpoints.get(key) or [])
        if collapse(str(p or ''))
    }


def predicate_contract_allows_endpoints(
    contract: dict[str, Any],
    *,
    subject_label: str = '',
    object_label: str = '',
) -> bool:
    """True when endpoint primaries satisfy the contract (empty allow-list = unconstrained)."""
    if not contract:
        return True
    subj_ok = _contract_endpoint_primaries(contract, 'subject')
    obj_ok = _contract_endpoint_primaries(contract, 'object')
    subj = collapse(subject_label or '').casefold()
    obj = collapse(object_label or '').casefold()
    if subj_ok and subj and subj not in subj_ok:
        return False
    if obj_ok and obj and obj not in obj_ok:
        return False
    return True


def _contract_evidence_cues(contract: dict[str, Any]) -> list[str]:
    evidence = contract.get('evidence') if isinstance(contract.get('evidence'), dict) else {}
    cues: list[str] = []
    seen: set[str] = set()
    for item in (evidence.get('predicate_cues_any') or []):
        cue = collapse(str(item or '')).casefold()
        if len(cue) < 2 or cue in seen:
            continue
        seen.add(cue)
        cues.append(cue)
    direction = contract.get('direction') if isinstance(contract.get('direction'), dict) else {}
    for bucket in ('forward', 'reverse'):
        for item in (direction.get('direction_cues') or {}).get(bucket) or [] if isinstance(
            direction.get('direction_cues'), dict
        ) else []:
            cue = collapse(str(item or '')).casefold()
            if len(cue) < 2 or cue in seen:
                continue
            seen.add(cue)
            cues.append(cue)
    role = direction.get('role_assignment') if isinstance(
        direction.get('role_assignment'), dict
    ) else {}
    for item in (role.get('verb_cues') or []):
        cue = collapse(str(item or '')).casefold()
        if len(cue) < 2 or cue in seen:
            continue
        seen.add(cue)
        cues.append(cue)
    agent_cue = collapse(str(direction.get('agent_cue') or '')).casefold()
    if len(agent_cue) >= 2 and agent_cue not in seen:
        cues.append(agent_cue)
    canon = collapse(str(contract.get('_canonical') or '')).casefold()
    if len(canon) >= 2 and canon not in seen:
        cues.insert(0, canon)
    return cues


@dataclass
class EvidenceWindow:
    """Bounded source window for scalable claim grounding."""
    kind: str
    start: int
    end: int
    text: str
    source_file: str | None = None
    parent_heading: str = ''


@dataclass
class DocumentEvidenceIndex:
    """Document-scoped sentence/table/section windows + surface occurrence map."""
    source_file: str | None
    text: str
    windows: list[EvidenceWindow] = field(default_factory=list)
    # surface.casefold() -> list of (start, end) absolute offsets
    surface_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def local_windows(
        self,
        *,
        chunk_start: int | None = None,
        chunk_end: int | None = None,
        kinds: list[str] | None = None,
    ) -> list[EvidenceWindow]:
        allowed = {k.casefold() for k in (kinds or []) if k} or None
        out: list[EvidenceWindow] = []
        for win in self.windows:
            if allowed is not None and win.kind.casefold() not in allowed:
                continue
            if chunk_start is not None and win.end <= int(chunk_start):
                continue
            if chunk_end is not None and win.start >= int(chunk_end):
                continue
            out.append(win)
        return out


def build_document_evidence_index(
    document_text: str,
    *,
    source_file: str | None = None,
    policy: Policy | None = None,
    entity_surfaces: list[str] | None = None,
) -> DocumentEvidenceIndex:
    """Build once per document; reuse for quote recovery and candidate inference."""
    pol = policy if policy is not None else load_policy()
    settings = _semantic_relation_recovery_settings(pol)
    doc = document_text or ''
    index = DocumentEvidenceIndex(source_file=source_file, text=doc)
    if not doc or not settings.get('evidence_index_enabled', True):
        return index

    wanted = {w.casefold() for w in settings.get('windows') or []}
    sentence_max = int(settings['sentence_max_chars'])
    table_max = int(settings['table_row_max_chars'])
    section_max = int(settings['section_block_max_chars'])

    if 'sentence' in wanted:
        # Split on sentence/line boundaries without domain regexes.
        cursor = 0
        for match in re.finditer(r'[^\n.!?]+(?:[.!?]+|\n+|$)', doc):
            raw = match.group(0)
            start = match.start()
            end = match.end()
            text = raw.strip()
            if len(text) < 8 or len(text) > sentence_max:
                cursor = end
                continue
            # Re-align stripped span inside the match.
            lead = len(raw) - len(raw.lstrip())
            abs_start = start + lead
            abs_end = abs_start + len(text)
            index.windows.append(EvidenceWindow(
                kind='sentence',
                start=abs_start,
                end=abs_end,
                text=doc[abs_start:abs_end],
                source_file=source_file,
            ))
            cursor = end
        del cursor

    if 'table_row' in wanted:
        for match in re.finditer(r'^\|.*\|$', doc, flags=re.MULTILINE):
            text = match.group(0).strip()
            if len(text) < 3 or len(text) > table_max:
                continue
            index.windows.append(EvidenceWindow(
                kind='table_row',
                start=match.start(),
                end=match.start() + len(match.group(0)),
                text=text,
                source_file=source_file,
            ))
        # HTML table rows when present.
        for match in re.finditer(r'<tr\b[^>]*>.*?</tr>', doc, flags=re.IGNORECASE | re.DOTALL):
            text = re.sub(r'<[^>]+>', ' ', match.group(0))
            text = collapse(text)
            if len(text) < 3 or len(text) > table_max:
                continue
            index.windows.append(EvidenceWindow(
                kind='table_row',
                start=match.start(),
                end=match.end(),
                text=doc[match.start():match.end()],
                source_file=source_file,
            ))

    if 'section_block' in wanted:
        # Reuse section-aware chunk spans (bounded) as section blocks.
        try:
            sections = section_aware_chunk_spans(doc, section_max)
        except Exception:
            sections = []
        if sections:
            for sec in sections:
                start = int(sec.get('start') or 0)
                end = int(sec.get('end') or 0)
                text = str(sec.get('text') or doc[start:end])
                if len(text.strip()) < 8:
                    continue
                index.windows.append(EvidenceWindow(
                    kind='section_block',
                    start=start,
                    end=end,
                    text=text,
                    source_file=source_file,
                    parent_heading=str(sec.get('parent_heading') or ''),
                ))
        else:
            # Fallback: consecutive non-empty paragraph blocks.
            for match in re.finditer(r'(?:.+\n?)+?(?:\n\s*\n|$)', doc):
                text = match.group(0).strip()
                if len(text) < 8 or len(text) > section_max:
                    continue
                index.windows.append(EvidenceWindow(
                    kind='section_block',
                    start=match.start(),
                    end=match.end(),
                    text=doc[match.start():match.end()],
                    source_file=source_file,
                ))

    folded = doc.casefold()
    for surface in entity_surfaces or []:
        key = collapse(str(surface or '')).casefold()
        if len(key) < 3:
            continue
        spans = index.surface_spans.setdefault(key, [])
        start = 0
        while True:
            idx = folded.find(key, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(key)))
            start = idx + max(1, len(key))
    return index


def _recover_quote_from_evidence_index(
    index: DocumentEvidenceIndex,
    *,
    subject: str,
    obj: str,
    predicate: str,
    policy: Policy | None = None,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
) -> str:
    """Shortest window quote that supports the claim; never invents text."""
    pol = policy if policy is not None else load_policy()
    contract = predicate_contract(predicate, policy=pol)
    evidence_cfg = contract.get('evidence') if isinstance(contract.get('evidence'), dict) else {}
    nearby = evidence_cfg.get('nearby_recovery') if isinstance(
        evidence_cfg.get('nearby_recovery'), dict
    ) else {}
    if contract and nearby.get('enabled') is False:
        return ''
    kinds = [
        collapse(str(k or '')).casefold()
        for k in (nearby.get('windows') or [])
        if collapse(str(k or ''))
    ]
    settings = _semantic_relation_recovery_settings(pol)
    prefer_local = bool(settings.get('prefer_local_windows', True))
    windows = index.local_windows(
        chunk_start=chunk_start if prefer_local else None,
        chunk_end=chunk_end if prefer_local else None,
        kinds=kinds or None,
    )
    if not windows:
        windows = index.windows
    candidates: list[str] = []
    for win in windows:
        quote = win.text
        if not quote or len(quote) > 1200:
            continue
        if _claim_supported_by_quote(
            quote,
            subject,
            obj,
            attribute_mode=False,
            predicate=predicate,
            policy=pol,
        ):
            candidates.append(quote.strip())
    if candidates:
        candidates.sort(key=len)
        return candidates[0]
    # Fall back to existing full-doc deterministic recovery.
    return _recover_claim_evidence_quote(
        index.text,
        subject=subject,
        obj=obj,
        predicate=predicate,
        attribute_mode=False,
        policy=pol,
    )


def _normalize_orientation_from_contract(
    quote: str,
    subject: str,
    obj: str,
    predicate: str,
    *,
    policy: Policy | None = None,
) -> tuple[str, str, str, bool]:
    """Flip endpoints when YAML orientation says patient precedes agent cue."""
    pol = policy if policy is not None else load_policy()
    contract = predicate_contract(predicate, policy=pol)
    if not contract:
        return subject, obj, predicate, False
    direction = contract.get('direction') if isinstance(contract.get('direction'), dict) else {}
    orientation = collapse(str(direction.get('orientation') or '')).casefold()
    if orientation != 'patient_before_agent_cue':
        return subject, obj, canonical_predicate(predicate, policy=pol) or predicate, False
    cue = collapse(str(direction.get('agent_cue') or '')).casefold()
    folded = (quote or '').casefold()
    if not cue or cue not in folded:
        return subject, obj, canonical_predicate(predicate, policy=pol) or predicate, False
    idx = folded.find(cue)
    before = folded[:idx]
    after = folded[idx + len(cue):]
    subj_before = _endpoint_supported_in_quote(subject, before, policy=pol)
    subj_after = _endpoint_supported_in_quote(subject, after, policy=pol)
    obj_before = _endpoint_supported_in_quote(obj, before, policy=pol)
    obj_after = _endpoint_supported_in_quote(obj, after, policy=pol)
    canon = canonical_predicate(predicate, policy=pol) or predicate
    # Already oriented: patient(subject) before cue, agent(object) after cue.
    if subj_before and obj_after:
        return subject, obj, canon, False
    # Swapped: agent currently in subject, patient in object → flip.
    if subj_after and obj_before:
        return obj, subject, canon, True
    return subject, obj, canon, False


def _entity_label_by_surface(
    bundle: ExtractionBundle,
    surface: str,
) -> str:
    key = collapse(surface or '').casefold()
    if not key:
        return ''
    for ent in bundle.entities or []:
        if not isinstance(ent, dict):
            continue
        if collapse(str(ent.get('text') or '')).casefold() == key:
            return collapse(str(ent.get('label') or ''))
        meta = ent.get('metadata') if isinstance(ent.get('metadata'), dict) else {}
        for ident in meta.get('identifiers') or []:
            if isinstance(ident, dict):
                val = collapse(str(ident.get('value') or ''))
            else:
                val = collapse(str(ident or ''))
            if val.casefold() == key:
                return collapse(str(ent.get('label') or ''))
    return ''


def _relation_fingerprint(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    return (
        collapse(subject or '').casefold(),
        collapse(predicate or '').casefold(),
        collapse(obj or '').casefold(),
    )


def recover_semantic_relations(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    document_text: str = '',
    source_file: str | None = None,
    evidence_index: DocumentEvidenceIndex | None = None,
) -> dict[str, int]:
    """Source-grounded semantic recovery: repair evidence/orientation + optional candidates.

    Never confirms from LLM alone. Re-runs claim verification and rebuilds
    confirmed/quarantine graphs when sync is enabled.
    """
    settings = _semantic_relation_recovery_settings(policy)
    stats = {
        'repaired': 0,
        'oriented': 0,
        'candidates_added': 0,
        'skipped': 0,
    }
    if not settings.get('enabled'):
        return stats
    doc = document_text or ''
    if not doc:
        return stats

    surfaces = [
        collapse(str(e.get('text') or ''))
        for e in (bundle.entities or [])
        if isinstance(e, dict) and collapse(str(e.get('text') or ''))
    ]
    index = evidence_index or build_document_evidence_index(
        doc,
        source_file=source_file,
        policy=policy,
        entity_surfaces=surfaces,
    )

    existing = {
        _relation_fingerprint(
            str(r.get('subject') or ''),
            str(r.get('predicate') or ''),
            str(r.get('object') or ''),
        )
        for r in (bundle.relations or [])
        if isinstance(r, dict)
    }
    max_repairs = int(settings['max_repairs_per_doc'])
    mutated = False

    for rel in list(bundle.relations or []):
        if stats['repaired'] + stats['oriented'] >= max_repairs:
            break
        if not isinstance(rel, dict):
            continue
        predicate = collapse(str(rel.get('predicate') or ''))
        subject = collapse(str(rel.get('subject') or ''))
        obj = collapse(str(rel.get('object') or ''))
        if not predicate or not subject or not obj:
            stats['skipped'] += 1
            continue
        # Hierarchy stays under existing gates — contracts do not rewrite them.
        if predicate.casefold() in hierarchy_predicate_set(policy):
            continue
        contract = predicate_contract(predicate, policy=policy)
        meta = rel.get('metadata') if isinstance(rel.get('metadata'), dict) else {}
        meta = dict(meta)
        status = collapse(str(meta.get('evidence_status') or '')).casefold()
        evidence = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else {}
        quote = str(evidence.get('source_text') or '')
        prov = meta.get('provenance') if isinstance(meta.get('provenance'), dict) else {}
        chunk_start = prov.get('chunk_start')
        chunk_end = prov.get('chunk_end')
        try:
            chunk_start_i = int(chunk_start) if chunk_start is not None else None
        except (TypeError, ValueError):
            chunk_start_i = None
        try:
            chunk_end_i = int(chunk_end) if chunk_end is not None else None
        except (TypeError, ValueError):
            chunk_end_i = None

        # Orientation repair when a quote exists.
        if quote:
            new_subj, new_obj, new_pred, flipped = _normalize_orientation_from_contract(
                quote, subject, obj, predicate, policy=policy,
            )
            if flipped:
                # Preserve original claim for audit.
                history = list(meta.get('recovery_history') or [])
                history.append({
                    'action': 'orientation_flip',
                    'from': {'subject': subject, 'predicate': predicate, 'object': obj},
                    'to': {'subject': new_subj, 'predicate': new_pred, 'object': new_obj},
                    'source_file': source_file,
                })
                meta['recovery_history'] = history
                rel['subject'] = new_subj
                rel['object'] = new_obj
                rel['predicate'] = new_pred
                subject, obj, predicate = new_subj, new_obj, new_pred
                stats['oriented'] += 1
                mutated = True

        needs_quote = (
            status in {'', 'missing', 'unverified'}
            or not quote
            or not bool((evidence or {}).get('claim_verified'))
        )
        if needs_quote:
            recovered = _recover_quote_from_evidence_index(
                index,
                subject=subject,
                obj=obj,
                predicate=predicate,
                policy=policy,
                chunk_start=chunk_start_i,
                chunk_end=chunk_end_i,
            )
            if recovered and recovered != quote:
                quote = recovered
                meta['evidence'] = {'source_text': recovered, 'source_file': source_file}
                meta['evidence_recovery'] = 'semantic_index_span'
                history = list(meta.get('recovery_history') or [])
                history.append({
                    'action': 'evidence_recover',
                    'quote': recovered[:240],
                    'source_file': source_file,
                })
                meta['recovery_history'] = history
                mutated = True
                stats['repaired'] += 1
            elif quote:
                # Re-check orientation on recovered-or-existing quote.
                new_subj, new_obj, new_pred, flipped = _normalize_orientation_from_contract(
                    quote, subject, obj, predicate, policy=policy,
                )
                if flipped:
                    history = list(meta.get('recovery_history') or [])
                    history.append({
                        'action': 'orientation_flip',
                        'from': {'subject': subject, 'predicate': predicate, 'object': obj},
                        'to': {'subject': new_subj, 'predicate': new_pred, 'object': new_obj},
                        'source_file': source_file,
                    })
                    meta['recovery_history'] = history
                    rel['subject'] = new_subj
                    rel['object'] = new_obj
                    rel['predicate'] = new_pred
                    subject, obj, predicate = new_subj, new_obj, new_pred
                    stats['oriented'] += 1
                    mutated = True

        # Contract endpoint type gate (soft: mark, do not invent).
        subj_label = _entity_label_by_surface(bundle, subject)
        obj_label = _entity_label_by_surface(bundle, obj)
        if contract and not predicate_contract_allows_endpoints(
            contract, subject_label=subj_label, object_label=obj_label,
        ):
            meta['predicate_contract_type_violation'] = True
        else:
            meta.pop('predicate_contract_type_violation', None)

        prev_eligible = meta.get('graph_eligible')
        prev_status = collapse(str(meta.get('evidence_status') or '')).casefold()
        meta.pop('graph_eligible', None)
        meta.pop('evidence_status', None)
        rel['metadata'] = _attach_claim_evidence(
            meta,
            doc,
            subject=subject,
            obj=obj,
            predicate=predicate,
            attribute_mode=False,
            source_file=source_file,
            policy=policy,
        )
        new_meta = rel['metadata'] if isinstance(rel.get('metadata'), dict) else {}
        new_status = collapse(str(new_meta.get('evidence_status') or '')).casefold()
        if (
            new_meta.get('graph_eligible') != prev_eligible
            or new_status != prev_status
        ):
            mutated = True
        if contract:
            rel['predicate'] = canonical_predicate(predicate, policy=policy) or predicate
            rel['relation_category'] = collapse(
                str(contract.get('category') or rel.get('relation_category') or 'semantic')
            )

    # Candidate generation from windows (contract cues + known endpoints only).
    max_candidates = int(settings['max_candidates_per_doc'])
    data = policy.data if isinstance(policy.data, dict) else {}
    contracts = data.get('predicate_contracts') if isinstance(
        data.get('predicate_contracts'), dict
    ) else {}
    entity_by_key: dict[str, dict[str, Any]] = {}
    for ent in bundle.entities or []:
        if not isinstance(ent, dict):
            continue
        key = collapse(str(ent.get('text') or '')).casefold()
        if key:
            entity_by_key[key] = ent

    if max_candidates > 0 and contracts:
        cand_max_chars = int(settings['candidate_max_window_chars'])
        cue_ctx = int(settings['candidate_cue_context_chars'])
        # Prefer longer cues first so "prepared by" beats bare "prepared".
        def _cue_slices(folded: str, cues: list[str]) -> list[tuple[str, int, int]]:
            hits: list[tuple[str, int, int]] = []
            for cue in sorted(cues, key=len, reverse=True):
                start = 0
                while True:
                    idx = folded.find(cue, start)
                    if idx < 0:
                        break
                    hits.append((cue, idx, idx + len(cue)))
                    start = idx + max(1, len(cue))
            return hits

        for canon, contract in contracts.items():
            if stats['candidates_added'] >= max_candidates:
                break
            if not isinstance(contract, dict):
                continue
            gen = contract.get('candidate_generation') if isinstance(
                contract.get('candidate_generation'), dict
            ) else {}
            if not gen.get('enabled'):
                continue
            pred = collapse(str(canon or ''))
            if not pred or pred.casefold() in hierarchy_predicate_set(policy):
                continue
            cues = _contract_evidence_cues({**contract, '_canonical': pred})
            if not cues:
                continue
            kinds = [
                collapse(str(k or '')).casefold()
                for k in (gen.get('sources') or ['sentence'])
                if collapse(str(k or ''))
            ]
            for win in index.local_windows(kinds=kinds or ['sentence']):
                if stats['candidates_added'] >= max_candidates:
                    break
                if len(win.text) > cand_max_chars:
                    continue
                folded = win.text.casefold()
                cue_hits = _cue_slices(folded, cues)
                if not cue_hits:
                    continue
                for cue, cue_lo, cue_hi in cue_hits:
                    if stats['candidates_added'] >= max_candidates:
                        break
                    # Tight cue-centered slice — never whole-document co-mention.
                    local_lo = max(0, cue_lo - cue_ctx)
                    local_hi = min(len(win.text), cue_hi + cue_ctx)
                    local_quote = win.text[local_lo:local_hi]
                    local_folded = local_quote.casefold()
                    hits = [
                        ent for key, ent in entity_by_key.items()
                        if key in local_folded
                    ]
                    if len(hits) < 2:
                        continue
                    for i, left in enumerate(hits):
                        for right in hits[i + 1:]:
                            if stats['candidates_added'] >= max_candidates:
                                break
                            for subject_ent, object_ent in (
                                (left, right),
                                (right, left),
                            ):
                                subj = collapse(str(subject_ent.get('text') or ''))
                                obj = collapse(str(object_ent.get('text') or ''))
                                fp = _relation_fingerprint(subj, pred, obj)
                                if fp in existing:
                                    continue
                                if not predicate_contract_allows_endpoints(
                                    {**contract, '_canonical': pred},
                                    subject_label=collapse(str(subject_ent.get('label') or '')),
                                    object_label=collapse(str(object_ent.get('label') or '')),
                                ):
                                    continue
                                if not _claim_supported_by_quote(
                                    local_quote,
                                    subj,
                                    obj,
                                    attribute_mode=False,
                                    predicate=pred,
                                    policy=policy,
                                ):
                                    continue
                                abs_start = win.start + local_lo
                                abs_end = win.start + local_hi
                                meta = {
                                    'extraction_method': 'semantic_relation_recovery',
                                    'evidence': {
                                        'source_text': local_quote.strip(),
                                        'source_file': source_file,
                                    },
                                    'evidence_recovery': 'semantic_candidate',
                                    'provenance': {
                                        'source_file': source_file,
                                        'chunk_start': abs_start,
                                        'chunk_end': abs_end,
                                        'chunk_method': win.kind,
                                        'parent_heading': win.parent_heading,
                                        'cue': cue,
                                    },
                                    'recovery_history': [{
                                        'action': 'candidate_from_window',
                                        'window_kind': win.kind,
                                        'cue': cue,
                                        'source_file': source_file,
                                    }],
                                }
                                meta = _attach_claim_evidence(
                                    meta,
                                    doc,
                                    subject=subj,
                                    obj=obj,
                                    predicate=pred,
                                    attribute_mode=False,
                                    source_file=source_file,
                                    policy=policy,
                                )
                                if meta.get('graph_eligible') is not True:
                                    continue
                                bundle.relations.append({
                                    'subject': subj,
                                    'predicate': pred,
                                    'object': obj,
                                    'confidence': 0.55,
                                    'metadata': meta,
                                    'relation_category': collapse(
                                        str(contract.get('category') or 'semantic')
                                    ),
                                    'subject_id': subject_ent.get('id') or (
                                        (subject_ent.get('metadata') or {}).get('canonical_id')
                                        if isinstance(subject_ent.get('metadata'), dict) else None
                                    ),
                                    'object_id': object_ent.get('id') or (
                                        (object_ent.get('metadata') or {}).get('canonical_id')
                                        if isinstance(object_ent.get('metadata'), dict) else None
                                    ),
                                    'subject_ref_kind': 'entity',
                                    'object_ref_kind': 'entity',
                                })
                                existing.add(fp)
                                stats['candidates_added'] += 1
                                mutated = True
                                break

    if mutated:
        recover_missing_endpoints(
            bundle, policy, document_text=doc, source_file=source_file,
        )
        bundle.warnings.extend(
            validate_extraction_graph(
                bundle, policy, source_file=source_file, document_text=doc,
            )
        )
        if settings.get('sync_quarantine_after_recovery', True):
            bundle.graph, bundle.quarantine = build_confirmed_graph(
                bundle, policy, source_file=source_file,
            )
    _diag_record('semantic_relation_recovery', **stats)
    return stats


def _key_occurrence_spans(folded_doc: str, keys: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for key in keys:
        if not key:
            continue
        start = 0
        while True:
            idx = folded_doc.find(key, start)
            if idx < 0:
                break
            end = idx + len(key)
            spans.append((idx, end))
            start = idx + max(1, len(key))
    return spans


def _membership_evidence_markers(policy: Policy | None = None) -> list[str]:
    """YAML membership / section-list cues used to ground hierarchy nearby recovery."""
    pol = policy if policy is not None else load_policy()
    settings = _claim_evidence_recovery_settings(pol)
    markers: list[str] = []
    if settings.get('reuse_section_membership_markers', True):
        markers.extend(_policy_labels(pol, 'hierarchy', 'section_membership_markers'))
    data = pol.data if isinstance(pol.data, dict) else {}
    assertion = data.get('assertion_classification') if isinstance(data.get('assertion_classification'), dict) else {}
    for item in assertion.get('membership_property_markers') or []:
        label = collapse(str(item or ''))
        if label:
            markers.append(label)
    out: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        fold = marker.casefold()
        if len(fold) < 3 or fold in seen:
            continue
        seen.add(fold)
        out.append(fold)
    return out


def _span_has_membership_evidence(span: str, *, policy: Policy | None = None) -> bool:
    folded = (span or '').casefold()
    if not folded:
        return False
    return any(marker in folded for marker in _membership_evidence_markers(policy))


def _object_keys_are_identifiers(obj_keys: list[str], *, policy: Policy | None = None) -> bool:
    """True when any object key is an identity code from YAML identifier patterns."""
    pol = policy if policy is not None else load_policy()
    settings = _endpoint_recovery_settings(pol)
    patterns = settings.get('identifier_patterns') or []
    for key in obj_keys or []:
        text = collapse(str(key or ''))
        if not text:
            continue
        if _match_identifier_kind(text, patterns):
            return True
        tags = _engineering_tags_in(text, policy=pol)
        if tags and any(_normalize_equipment_tag(t) == _normalize_equipment_tag(text) for t in tags):
            return True
    return False


def _span_has_table_context_for_object(
    span: str,
    obj_keys: list[str],
    *,
    policy: Policy | None = None,
) -> bool:
    """True when an object key sits on a pipe/HTML table row inside the span."""
    settings = _claim_evidence_recovery_settings(policy)
    min_cells = int(settings.get('min_pipe_cells') or 2)
    for line in (span or '').splitlines():
        folded = line.casefold()
        if not any(key in folded for key in obj_keys if key):
            continue
        cells = _split_pipe_row(line)
        if cells is not None and len(cells) >= min_cells:
            return True
        if '<td' in folded or '<th' in folded or '<tr' in folded:
            return True
    return False


def _line_span_at(text: str, offset: int) -> tuple[int, int]:
    if offset < 0:
        return (0, 0)
    start = text.rfind('\n', 0, offset) + 1
    end = text.find('\n', offset)
    if end < 0:
        end = len(text)
    return start, end


def _nearby_section_ownership_ok(
    document_text: str,
    span_lo: int,
    span_hi: int,
    *,
    subject: str,
    subj_keys: list[str],
    obj_keys: list[str],
    predicate: str = '',
    policy: Policy | None = None,
) -> bool:
    """Subject owns the object inside the candidate span (section index or local)."""
    doc = document_text or ''
    if not doc or span_lo < 0 or span_hi <= span_lo:
        return False
    span = doc[span_lo:span_hi]
    folded_span = span.casefold()

    # Prefer equipment-section index: subject matches the section covering object.
    sections = build_equipment_section_index(doc, policy=policy)
    for o0, _o1 in _key_occurrence_spans(folded_span, obj_keys):
        abs_off = span_lo + o0
        sec = section_at_offset(sections, abs_off)
        if sec and _subject_matches_section(subject, sec, policy=policy):
            return True

    # Local ownership: subject before object; not peer cells on the same table row.
    subj_rel = -1
    for key in subj_keys:
        pos = folded_span.find(key)
        if pos >= 0 and (subj_rel < 0 or pos < subj_rel):
            subj_rel = pos
    if subj_rel < 0:
        return False
    obj_rel = -1
    for key in obj_keys:
        pos = folded_span.find(key, subj_rel + 1)
        if pos >= 0 and (obj_rel < 0 or pos < obj_rel):
            obj_rel = pos
    if obj_rel < 0:
        return False

    parent_markers = {
        collapse(p).casefold()
        for p in _policy_labels(policy if policy is not None else load_policy(), 'hierarchy', 'parent_as_subject_markers')
    }
    pred = collapse(predicate or '').casefold()
    if pred in parent_markers or pred in hierarchy_predicate_set(policy if policy is not None else load_policy()):
        if subj_rel >= obj_rel:
            return False

    subj_line = span[_line_span_at(span, subj_rel)[0]:_line_span_at(span, subj_rel)[1]]
    obj_line = span[_line_span_at(span, obj_rel)[0]:_line_span_at(span, obj_rel)[1]]
    if collapse(subj_line) == collapse(obj_line):
        cells = _split_pipe_row(subj_line)
        if cells is not None and len(cells) >= 2:
            # Same table row ⇒ peer cells, not owner/member unless section index matched.
            return False
    return True


def _nearby_hierarchy_span_valid(
    document_text: str,
    span_lo: int,
    span_hi: int,
    *,
    subject: str,
    subj_keys: list[str],
    obj_keys: list[str],
    predicate: str = '',
    policy: Policy | None = None,
) -> bool:
    """Hierarchy nearby verify: endpoints + membership + section owner + table context."""
    doc = document_text or ''
    span = doc[span_lo:span_hi]
    settings = _claim_evidence_recovery_settings(policy)
    if settings.get('require_membership_evidence', True) and not _span_has_membership_evidence(span, policy=policy):
        return False
    need_table = bool(settings.get('require_table_context', True))
    if need_table and settings.get('require_table_context_for_identifier_objects', True):
        need_table = _object_keys_are_identifiers(obj_keys, policy=policy)
    if need_table and not _span_has_table_context_for_object(
        span, obj_keys, policy=policy,
    ):
        return False
    if settings.get('require_section_ownership', True) and not _nearby_section_ownership_ok(
        doc,
        span_lo,
        span_hi,
        subject=subject,
        subj_keys=subj_keys,
        obj_keys=obj_keys,
        predicate=predicate,
        policy=policy,
    ):
        return False
    return True


def _recover_nearby_endpoint_span(
    document_text: str,
    subj_keys: list[str],
    obj_keys: list[str],
    *,
    max_chars: int,
    subject: str = '',
    predicate: str = '',
    policy: Policy | None = None,
    enforce_hierarchy_gates: bool = False,
) -> str:
    """Shortest contiguous source substring containing both endpoint keys.

    Used when subject/object co-occur nearby but not on one line (e.g. heading +
    table row). Hierarchy predicates also require membership evidence, section
    ownership, and table context. Returns verbatim doc slice only; never invents text.
    """
    doc = document_text or ''
    if not doc or max_chars < 4 or not subj_keys or not obj_keys:
        return ''
    folded = doc.casefold()
    subj_spans = _key_occurrence_spans(folded, subj_keys)
    obj_spans = _key_occurrence_spans(folded, obj_keys)
    if not subj_spans or not obj_spans:
        return ''
    candidates: list[tuple[int, int]] = []
    for s0, s1 in subj_spans:
        for o0, o1 in obj_spans:
            lo, hi = (s0, max(s1, o1)) if s0 <= o0 else (o0, max(o1, s1))
            # Keep full lines so pipe/HTML table rows stay intact for table_context.
            line_lo, _ = _line_span_at(doc, lo)
            _, line_hi = _line_span_at(doc, max(lo, hi - 1))
            lo, hi = line_lo, line_hi
            width = hi - lo
            if width <= 0 or width > max_chars:
                continue
            if enforce_hierarchy_gates and not _nearby_hierarchy_span_valid(
                doc,
                lo,
                hi,
                subject=subject,
                subj_keys=subj_keys,
                obj_keys=obj_keys,
                predicate=predicate,
                policy=policy,
            ):
                continue
            candidates.append((lo, hi))
    if not candidates:
        return ''
    candidates.sort(key=lambda span: (span[1] - span[0], span[0]))
    best_lo, best_hi = candidates[0]
    near = [
        c for c in candidates
        if (c[1] - c[0]) <= (best_hi - best_lo) + 24
    ]
    if len({doc[c[0]:c[1]].casefold() for c in near}) > 3:
        return ''
    return doc[best_lo:best_hi]


def _recover_claim_evidence_quote(
    document_text: str,
    *,
    subject: str,
    obj: str,
    predicate: str = '',
    attribute_mode: bool = False,
    policy: Policy | None = None,
) -> str:
    """Deterministic evidence recovery: shortest source span grounding endpoints.

    Prefers a single supportive line; falls back to a short contiguous nearby
    span when endpoints co-occur close together across lines. Hierarchy nearby
    recovery requires membership + section ownership + table context. Does not
    invent text. Returns '' when no unique supportive span is found.
    """
    doc = document_text or ''
    if not doc or not subject or not obj:
        return ''
    pol = policy if policy is not None else load_policy()
    subj_tags = [_normalize_equipment_tag(t) for t in _engineering_tags_in(subject, policy=pol)]
    obj_tags = [_normalize_equipment_tag(t) for t in _engineering_tags_in(obj, policy=pol)]
    subj_keys = [t for t in subj_tags if t] or ([collapse(subject).casefold()] if len(collapse(subject)) >= 4 else [])
    if attribute_mode:
        obj_keys = [
            collapse(c).casefold()
            for c in _attribute_value_candidates(obj, policy=pol)
            if collapse(c)
        ]
    else:
        obj_keys = [t for t in obj_tags if t] or ([collapse(obj).casefold()] if len(collapse(obj)) >= 4 else [])
    if not subj_keys or not obj_keys:
        return ''
    pred = collapse(predicate or '').casefold()
    enforce = (not attribute_mode) and pred in hierarchy_predicate_set(pol)
    pred_tokens = [t for t in _token_parts(predicate) if len(t) >= 4]
    candidates: list[str] = []
    # Line-oriented search keeps spans tight and source-faithful.
    for line in doc.splitlines():
        raw = line.strip()
        if len(raw) < 4 or len(raw) > 600:
            continue
        folded = raw.casefold()
        unesc = _unescape_source_markup(raw).casefold()
        if not any(k in folded or k in unesc for k in subj_keys):
            continue
        if not any(k in folded or k in unesc for k in obj_keys):
            continue
        if attribute_mode or not pred_tokens or any(t in folded or t in unesc for t in pred_tokens):
            if enforce:
                idx = doc.find(raw)
                if idx < 0:
                    idx = doc.casefold().find(raw.casefold())
                if idx < 0 or not _nearby_hierarchy_span_valid(
                    doc,
                    idx,
                    idx + len(raw),
                    subject=subject,
                    subj_keys=subj_keys,
                    obj_keys=obj_keys,
                    predicate=predicate,
                    policy=pol,
                ):
                    continue
            candidates.append(raw)
    if candidates:
        # Prefer shortest unique supportive line.
        candidates.sort(key=len)
        best = candidates[0]
        if sum(1 for c in candidates if c.casefold() == best.casefold()) > 1 and len(candidates) > 1:
            # Ambiguous identical lines are ok; different equal-length alternatives → skip.
            alts = [c for c in candidates if abs(len(c) - len(best)) <= 8]
            if len({c.casefold() for c in alts}) > 3:
                return ''
        return best
    # Nearby contiguous span (heading + membership/table body, etc.).
    nearby = _recover_nearby_endpoint_span(
        doc,
        subj_keys,
        obj_keys,
        max_chars=_claim_nearby_span_max_chars(pol),
        subject=subject,
        predicate=predicate,
        policy=pol,
        enforce_hierarchy_gates=enforce,
    )
    if nearby:
        return nearby
    if not attribute_mode:
        return ''
    pred_tokens = [t for t in _token_parts(predicate) if len(t) >= 4]
    label_lines: list[str] = []
    for line in doc.splitlines():
        raw = line.strip()
        if len(raw) < 4 or len(raw) > 600:
            continue
        folded = raw.casefold()
        unesc = _unescape_source_markup(raw).casefold()
        if not any(k in folded or k in unesc for k in obj_keys):
            continue
        if pred_tokens and any(t in folded or t in unesc for t in pred_tokens):
            label_lines.append(raw)
            continue
        parsed = _split_kv_segment(raw, separators=_kv_separators(pol))
        if parsed and pred_tokens:
            label, _sep, _val = parsed
            label_parts = _token_parts(_strip_label_unit_suffix(label))
            if label_parts & set(pred_tokens):
                label_lines.append(raw)
    if not label_lines:
        return ''
    label_lines.sort(key=len)
    return label_lines[0]


def _hierarchy_has_cycle(edges: list[tuple[str, str]]) -> list[str]:
    """Return nodes involved in cycles using parent→child edges."""
    graph: dict[str, list[str]] = {}
    for parent, child in edges:
        graph.setdefault(parent, []).append(child)
    visiting: set[str] = set()
    visited: set[str] = set()
    cyclic: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        visiting.add(node)
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt in visiting:
                cyclic.update(stack[stack.index(nxt):] + [nxt])
            elif nxt not in visited:
                dfs(nxt, stack)
        stack.pop()
        visiting.discard(node)
        visited.add(node)

    for node in list(graph):
        if node not in visited:
            dfs(node, [])
    return sorted(cyclic)


def load_prior_graph_entities(*, exclude_source: str | None = None) -> list[dict]:
    """Load entities from prior outputs/*_extraction.json when available (no registry assumed)."""
    prior: list[dict] = []
    if not OUTPUT_DIR.is_dir():
        return prior
    for path in sorted(OUTPUT_DIR.glob('*_extraction.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        source = collapse(str(data.get('source') or path.name))
        if exclude_source and source.casefold() == collapse(exclude_source).casefold():
            continue
        for entity in data.get('entities') or []:
            if isinstance(entity, dict):
                row = dict(entity)
                row['_prior_source'] = source
                prior.append(row)
    return prior

# ---------------------------------------------------------------------------
# 11) Graph validation — post-extract consistency warnings
# ---------------------------------------------------------------------------
def _endpoint_recovery_settings(policy: Policy | None = None) -> dict[str, Any]:
    """Load identity_resolution.endpoint_recovery (YAML-owned; no Python regex)."""
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    identity = data.get('identity_resolution') if isinstance(data.get('identity_resolution'), dict) else {}
    cfg = identity.get('endpoint_recovery') if isinstance(identity.get('endpoint_recovery'), dict) else {}
    kind_primary: dict[str, str] = {}
    raw_kinds = cfg.get('identifier_kind_primary') if isinstance(cfg.get('identifier_kind_primary'), dict) else {}
    for kind, primary in raw_kinds.items():
        k = collapse(str(kind or '')).casefold()
        p = collapse(str(primary or ''))
        if k and p:
            kind_primary[k] = p
    kind_subtype: dict[str, str] = {}
    raw_sub = cfg.get('identifier_kind_subtype') if isinstance(cfg.get('identifier_kind_subtype'), dict) else {}
    for kind, subtype in raw_sub.items():
        k = collapse(str(kind or '')).casefold()
        s = collapse(str(subtype or ''))
        if k and s:
            kind_subtype[k] = s
    # Middle-segment → node_catalog typing (resolved once from graph_schema).
    middle_typing: dict[str, tuple[str, str]] = {}
    schema = data.get('graph_schema') if isinstance(data.get('graph_schema'), dict) else {}
    node_catalog = schema.get('node_catalog') if isinstance(schema.get('node_catalog'), dict) else {}
    raw_middle = (
        cfg.get('engineering_tag_middle_node_catalog')
        if isinstance(cfg.get('engineering_tag_middle_node_catalog'), dict)
        else {}
    )
    for segment, node_key in raw_middle.items():
        seg = collapse(str(segment or '')).upper()
        key = collapse(str(node_key or ''))
        entry = node_catalog.get(key) if key else None
        if not seg or not isinstance(entry, dict):
            continue
        primary = collapse(str(entry.get('primary_type') or ''))
        subtype = collapse(str(entry.get('subtype') or ''))
        if not subtype:
            subs = entry.get('subtypes') if isinstance(entry.get('subtypes'), list) else []
            if len(subs) == 1:
                subtype = collapse(str(subs[0] or ''))
        if primary:
            middle_typing[seg] = (primary, subtype)
    pred_roles: dict[str, dict[str, str]] = {}
    raw_preds = (
        cfg.get('predicate_endpoint_primaries')
        if isinstance(cfg.get('predicate_endpoint_primaries'), dict)
        else {}
    )
    for pred, roles in raw_preds.items():
        if not isinstance(roles, dict):
            continue
        key = collapse(str(pred or '')).casefold()
        if not key:
            continue
        cleaned: dict[str, str] = {}
        for role, primary in roles.items():
            r = collapse(str(role or '')).casefold()
            p = collapse(str(primary or ''))
            if r in {'subject', 'object'} and p:
                cleaned[r] = p
        if cleaned:
            pred_roles[key] = cleaned
    patterns: list[dict[str, str]] = []
    for row in identity.get('identifier_patterns') or []:
        if not isinstance(row, dict):
            continue
        pattern = str(row.get('pattern') or '').strip()
        kind = collapse(str(row.get('kind') or ''))
        if pattern and kind:
            patterns.append({'pattern': pattern, 'kind': kind})
    literal_contracts: dict[str, dict[str, Any]] = {}
    raw_literal = (
        cfg.get('predicate_literal_object')
        if isinstance(cfg.get('predicate_literal_object'), dict)
        else {}
    )
    for pred, spec in raw_literal.items():
        if not isinstance(spec, dict):
            continue
        key = collapse(str(pred or '')).casefold()
        if not key:
            continue
        category = collapse(str(spec.get('category') or 'semantic')).casefold() or 'semantic'
        kinds = [
            collapse(str(k or '')).casefold()
            for k in (spec.get('allowed_literal_kinds') or [])
            if collapse(str(k or ''))
        ]
        subjects = [
            collapse(str(s or ''))
            for s in (spec.get('allowed_subject_primaries') or [])
            if collapse(str(s or ''))
        ]
        if not kinds:
            continue
        literal_contracts[key] = {
            'category': category,
            'allowed_literal_kinds': kinds,
            'allowed_subject_primaries': subjects,
        }
    return {
        'enabled': bool(cfg.get('enabled', False)),
        'require_verbatim_in_evidence': bool(cfg.get('require_verbatim_in_evidence', True)),
        'allow_prefix_alias': bool(cfg.get('allow_prefix_alias', False)),
        'identifier_kind_primary': kind_primary,
        'identifier_kind_subtype': kind_subtype,
        'engineering_tag_middle_typing': middle_typing,
        'predicate_endpoint_primaries': pred_roles,
        'identifier_patterns': patterns,
        'predicate_literal_object': literal_contracts,
    }


def _match_identifier_kind(surface: str, patterns: list[dict[str, str]]) -> str:
    """Return identifier pattern kind when surface fully matches a YAML pattern."""
    text = collapse(surface or '')
    if not text:
        return ''
    for row in patterns or []:
        compiled = _compile_regex(row.get('pattern') or '')
        if compiled is None:
            continue
        if compiled.fullmatch(text):
            return collapse(str(row.get('kind') or ''))
    return ''


def _evidence_quote_for_row(row: dict) -> str:
    meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
    evidence = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else None
    if evidence:
        return collapse(str(evidence.get('source_text') or ''))
    # Quarantine / graph rows may store evidence at top level.
    top = row.get('evidence') if isinstance(row.get('evidence'), dict) else None
    if top:
        return collapse(str(top.get('source_text') or ''))
    return ''


def _section_context_for_row(row: dict) -> dict[str, Any]:
    meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
    section = meta.get('section_context') if isinstance(meta.get('section_context'), dict) else None
    if section:
        return section
    top = row.get('section_context') if isinstance(row.get('section_context'), dict) else None
    return top if isinstance(top, dict) else {}


def _provenance_support_texts(meta: dict[str, Any] | None) -> list[str]:
    """Evidence + section_context surfaces used to ground unresolved subjects."""
    out: list[str] = []
    payload = meta if isinstance(meta, dict) else {}
    evidence = payload.get('evidence') if isinstance(payload.get('evidence'), dict) else {}
    section = payload.get('section_context') if isinstance(payload.get('section_context'), dict) else {}
    for src in (evidence, section):
        text = collapse(str(src.get('source_text') or ''))
        if text:
            out.append(text)
    for tag in section.get('equipment_tags') or []:
        value = collapse(str(tag or ''))
        if value:
            out.append(value)
    return out


def _subject_grounded_in_provenance(
    subject: str,
    meta: dict[str, Any] | None,
    *,
    policy: Policy | None = None,
) -> bool:
    """True when subject appears in evidence/section_context (exact, fuzzy, or tag)."""
    surface = collapse(subject or '')
    if not surface:
        return False
    payload = meta if isinstance(meta, dict) else {}
    for text in _provenance_support_texts(payload):
        if _object_verbatim_in_evidence(surface, text):
            return True
        # Owner cell may be a shorter prefix of a split-row subject
        # (e.g. "22-PK-3904: Blenders Blowing" vs "... Blowing system").
        if _object_verbatim_in_evidence(text, surface):
            return True
    section = payload.get('section_context') if isinstance(payload.get('section_context'), dict) else {}
    subj_tags = {
        _normalize_equipment_tag(t)
        for t in _engineering_tags_in(surface, policy=policy)
        if _normalize_equipment_tag(t)
    }
    sec_tags = {
        _normalize_equipment_tag(t)
        for t in (section.get('equipment_tags') or [])
        if collapse(str(t or ''))
    }
    if not sec_tags:
        sec_tags = {
            _normalize_equipment_tag(t)
            for t in _engineering_tags_in(str(section.get('source_text') or ''), policy=policy)
            if _normalize_equipment_tag(t)
        }
    if subj_tags and sec_tags and any(
        _tags_equivalent(a, b) for a in subj_tags for b in sec_tags
    ):
        return True
    return False


def _entity_text_surface(entity: dict[str, Any] | None) -> str:
    if not isinstance(entity, dict):
        return ''
    return collapse(str(entity.get('text') or entity.get('name') or ''))


def _resolve_endpoint_via_provenance(
    value: str,
    meta: dict[str, Any] | None,
    *,
    by_surface: dict[str, dict],
    index: dict[str, Any] | None = None,
    policy: Policy | None = None,
) -> tuple[str | None, str]:
    """Resolve subject via evidence/section_context without binding shared member tags.

    Exact/fuzzy owner-cell surfaces and *unique* engineering tags only. Shared
    tags on compound package entities must not steal row-group owners.
    """
    eid, kind = _resolve_endpoint_ref(
        value, by_surface=by_surface, index=index, policy=policy,
    )
    if kind == 'entity':
        # Accept only when the hit's canonical text matches the subject (exact/fuzzy).
        # Reject compound packages that merely list the subject's tag as a member id.
        ent = (index or {}).get('by_id', {}).get(eid or '') if index else None
        if ent is None:
            ent = by_surface.get(collapse(value).casefold())
        canon = _entity_text_surface(ent)
        if canon and (
            collapse(value).casefold() == canon.casefold()
            or _object_verbatim_in_evidence(value, canon)
            or _object_verbatim_in_evidence(canon, value)
        ):
            return eid, kind
        # Fall through to provenance owner resolution / recovery mint.
    if not _subject_grounded_in_provenance(value, meta, policy=policy):
        return None, 'unresolved'

    payload = meta if isinstance(meta, dict) else {}
    section = payload.get('section_context') if isinstance(payload.get('section_context'), dict) else {}
    owner = collapse(str(section.get('source_text') or ''))
    by_id = (index or {}).get('by_id') if isinstance((index or {}).get('by_id'), dict) else {}
    by_unique = (index or {}).get('by_unique_tag') if isinstance(
        (index or {}).get('by_unique_tag'), dict
    ) else {}

    def _eid_of(ent: dict[str, Any] | None) -> str | None:
        if not isinstance(ent, dict):
            return None
        return collapse(str(ent.get('id') or (ent.get('metadata') or {}).get('canonical_id') or '')) or None

    # 1) Exact owner-cell surface → entity text / alias / identifier key.
    if owner:
        hit = by_surface.get(owner.casefold())
        if hit is not None:
            canon = _entity_text_surface(hit)
            if canon.casefold() == owner.casefold() or _object_verbatim_in_evidence(owner, canon):
                eid2 = _eid_of(hit)
                if eid2:
                    return eid2, 'entity'

    # 2) Unique engineering tags only when entity *text* matches owner/subject.
    # Never bind a compound package that merely lists the tag as a member id.
    needles = [value]
    if owner:
        needles.append(owner)

    def _text_matches_needles(canon: str) -> bool:
        if not canon:
            return False
        return any(
            collapse(n).casefold() == canon.casefold()
            or _object_verbatim_in_evidence(n, canon)
            or _object_verbatim_in_evidence(canon, n)
            for n in needles
            if collapse(n or '')
        )

    tag_candidates = [
        collapse(str(t or ''))
        for t in (section.get('equipment_tags') or [])
        if collapse(str(t or ''))
    ]
    if owner:
        tag_candidates.extend(_engineering_tags_in(owner, policy=policy))
    tag_candidates.extend(_engineering_tags_in(value, policy=policy))
    seen_tags: set[str] = set()
    for tag in tag_candidates:
        norm = _normalize_equipment_tag(tag)
        if not norm or norm in seen_tags:
            continue
        seen_tags.add(norm)
        ent = by_unique.get(norm)
        eid2 = _eid_of(ent)
        if eid2 and _text_matches_needles(_entity_text_surface(ent)):
            return eid2, 'entity'

    # 3) Fuzzy unique match against entity *texts* only (not identifier index keys).
    fuzzy_ids: list[str] = []
    for ent in by_id.values():
        canon = _entity_text_surface(ent)
        if not _text_matches_needles(canon):
            continue
        eid3 = _eid_of(ent)
        if eid3 and eid3 not in fuzzy_ids:
            fuzzy_ids.append(eid3)
    if len(fuzzy_ids) == 1:
        return fuzzy_ids[0], 'entity'
    return None, 'unresolved'


def _engineering_tag_middle_segments(
    surface: str,
    *,
    policy: Policy | None = None,
) -> list[str]:
    """Middle letter-group of tags shaped like ``NN-XX-NNNN`` (policy patterns)."""
    out: list[str] = []
    for tag in _engineering_tags_in(surface, policy=policy):
        parts = [p for p in collapse(str(tag or '')).split('-') if p]
        if len(parts) >= 3 and parts[1].isalpha():
            mid = parts[1].upper()
            if mid not in out:
                out.append(mid)
    return out


def _infer_recovered_typing(
    surface: str,
    *,
    predicate: str,
    role: str,
    settings: dict[str, Any],
    policy: Policy | None = None,
) -> tuple[str, str]:
    """YAML-implied (primary, subtype). Empty primary → do not recover as entity.

    Single engineering_tag pattern; optional middle-segment → node_catalog override
    runs before the default kind→primary map.
    """
    pred = collapse(predicate or '').casefold()
    role_key = collapse(role or '').casefold()
    pred_map = settings.get('predicate_endpoint_primaries') or {}
    roles = pred_map.get(pred) if isinstance(pred_map, dict) else None
    if isinstance(roles, dict):
        primary = collapse(str(roles.get(role_key) or ''))
        if primary:
            return primary, ''
    middle_typing = settings.get('engineering_tag_middle_typing') or {}
    if isinstance(middle_typing, dict) and middle_typing:
        for mid in _engineering_tag_middle_segments(surface, policy=policy):
            hit = middle_typing.get(mid)
            if isinstance(hit, tuple) and hit and collapse(str(hit[0] or '')):
                return collapse(str(hit[0])), collapse(str(hit[1] if len(hit) > 1 else '') or '')
    kind_map = settings.get('identifier_kind_primary') or {}
    subtype_map = settings.get('identifier_kind_subtype') or {}
    kind = _match_identifier_kind(surface, settings.get('identifier_patterns') or [])
    if kind:
        primary = collapse(str(kind_map.get(kind.casefold()) or ''))
        if primary:
            return primary, collapse(str(subtype_map.get(kind.casefold()) or ''))
    # TAG: Title / embedded identifier — first YAML pattern with a primary mapping.
    for row in settings.get('identifier_patterns') or []:
        if not isinstance(row, dict):
            continue
        compiled = _compile_regex(row.get('pattern') or '')
        if compiled is None:
            continue
        if not compiled.search(collapse(surface or '')):
            continue
        kind_name = collapse(str(row.get('kind') or '')).casefold()
        if not kind_name:
            continue
        primary = collapse(str(kind_map.get(kind_name) or ''))
        if primary:
            return primary, collapse(str(subtype_map.get(kind_name) or ''))
    return '', ''


def _infer_recovered_primary(
    surface: str,
    *,
    predicate: str,
    role: str,
    settings: dict[str, Any],
    policy: Policy | None = None,
) -> str:
    """Unique YAML-implied primary, else empty (ambiguous → do not recover)."""
    primary, _subtype = _infer_recovered_typing(
        surface, predicate=predicate, role=role, settings=settings, policy=policy,
    )
    return primary


def _identity_reject_noise_surfaces(policy: Policy | None = None) -> set[str]:
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    identity = data.get('identity_resolution') if isinstance(data.get('identity_resolution'), dict) else {}
    return {
        collapse(str(x or '')).casefold()
        for x in (identity.get('reject_noise_surfaces') or [])
        if collapse(str(x or ''))
    }


def _surface_is_property_not_entity(
    surface: str,
    *,
    policy: Policy | None = None,
    predicate: str = '',
    evidence_quote: str = '',
    document_text: str = '',
    attribute_object: bool = False,
) -> bool:
    """True when surface is a scalar/property value, not an independent entity node.

    Uses non_entity_policy.properties_not_nodes intent via existing quantity /
    attribute detectors — no token blacklists of domain values.
    """
    text = collapse(surface or '')
    if not text:
        return True
    settings = _endpoint_recovery_settings(policy)
    # Identifier-shaped or tagged owner cells are entity candidates.
    if _match_identifier_kind(text, settings.get('identifier_patterns') or []):
        return False
    if _engineering_tags_in(text, policy=policy):
        return False
    if attribute_object:
        return True
    if text.casefold() in _identity_reject_noise_surfaces(policy):
        return True
    if _object_parses_as_quantity(text, policy=policy):
        return True
    if _object_looks_like_numeric_measurement(text, policy=policy):
        return True
    if predicate and _looks_like_technical_attribute_claim(
        predicate,
        text,
        evidence_quote=evidence_quote,
        document_text=document_text,
        policy=policy,
    ):
        return True
    return False


def _validate_recovered_entity_proposal(
    surface: str,
    *,
    proposed_label: str = '',
    predicate: str = '',
    role: str = 'subject',
    policy: Policy | None = None,
    evidence_quote: str = '',
    document_text: str = '',
    attribute_object: bool = False,
) -> tuple[str, str] | None:
    """Accept recovered node only with YAML-validated primary (LLM hint is not enough)."""
    text = collapse(surface or '')
    if not text:
        return None
    if _surface_is_property_not_entity(
        text,
        policy=policy,
        predicate=predicate,
        evidence_quote=evidence_quote,
        document_text=document_text,
        attribute_object=attribute_object,
    ):
        return None
    settings = _endpoint_recovery_settings(policy)
    if not _surface_is_atomic_endpoint(text, settings):
        return None
    primary, subtype = _infer_recovered_typing(
        text, predicate=predicate, role=role, settings=settings, policy=policy,
    )
    if not primary:
        # LLM / caller hint alone is insufficient.
        return None
    allowed = {n.casefold(): n for n in _primary_type_names(policy or load_policy())}
    if primary.casefold() not in allowed:
        return None
    # If a hint was supplied, it must agree with the independently inferred primary.
    hint = collapse(proposed_label or '')
    if hint and hint.casefold() != primary.casefold() and hint.casefold() in allowed:
        # Prefer YAML inference over conflicting LLM / recovery hint.
        pass
    return allowed[primary.casefold()], subtype


def _scrub_invalid_recovered_entities(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    document_text: str = '',
) -> list[str]:
    """Drop invalid recovered nodes; retarget attribute objects to literals.

    Returns removed entity ids. Preserves evidence / section_context on claims.
    """
    removed: list[str] = []
    kept: list[dict] = []
    attr_preds = {p.casefold() for p in attribute_predicate_hints(policy)}
    for ent in bundle.entities or []:
        if not isinstance(ent, dict):
            continue
        meta = ent.get('metadata') if isinstance(ent.get('metadata'), dict) else {}
        method = collapse(str(meta.get('extraction_method') or '')).casefold()
        if method not in {'llm_quarantine_resolver', 'endpoint_recovery'}:
            kept.append(ent)
            continue
        surface = collapse(str(ent.get('text') or ''))
        eid = collapse(str(ent.get('id') or meta.get('canonical_id') or ''))
        validated = _validate_recovered_entity_proposal(
            surface,
            proposed_label=collapse(str(ent.get('label') or '')),
            policy=policy,
            document_text=document_text,
            attribute_object=False,
        )
        if validated is None:
            if eid:
                removed.append(eid)
            continue
        primary, subtype = validated
        ent['label'] = primary
        meta = dict(meta)
        if subtype:
            meta['subtype'] = subtype
        meta['typing_reason'] = meta.get('typing_reason') or 'yaml_endpoint_recovery'
        ent['metadata'] = meta
        kept.append(ent)
    if not removed:
        bundle.entities = kept
        return removed
    bundle.entities = kept
    removed_set = set(removed)

    def _retarget(row: dict) -> None:
        if not isinstance(row, dict):
            return
        obj_id = collapse(str(row.get('object_id') or ''))
        subj_id = collapse(str(row.get('subject_id') or ''))
        if subj_id in removed_set:
            row['subject_id'] = None
            row['subject_ref_kind'] = 'unresolved'
        if obj_id not in removed_set:
            return
        obj = collapse(str(row.get('object') or ''))
        pred = collapse(str(row.get('predicate') or ''))
        meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
        quote = ''
        if isinstance(meta.get('evidence'), dict):
            quote = str(meta['evidence'].get('source_text') or '')
        is_attr = (
            pred.casefold() in attr_preds
            or _is_generic_attribute_predicate(pred, policy=policy)
            or collapse(str(row.get('relation_category') or '')).casefold() == 'attribute'
            or _looks_like_technical_attribute_claim(
                pred, obj, evidence_quote=quote, document_text=document_text, policy=policy,
            )
        )
        if is_attr:
            _apply_object_literal(
                row,
                obj=obj,
                category='attribute',
                typing_reason='property_not_entity',
            )
        else:
            row['object_id'] = None
            row['object_ref_kind'] = 'unresolved'
            row['object_value_kind'] = 'unresolved'

    for rel in bundle.relations or []:
        _retarget(rel)
    for trip in bundle.triplets or []:
        _retarget(trip)
    return removed


def _surface_is_atomic_endpoint(surface: str, settings: dict[str, Any]) -> bool:
    """True for identifier-shaped or short non-prose endpoint mentions."""
    text = collapse(surface or '')
    if not text:
        return False
    if _match_identifier_kind(text, settings.get('identifier_patterns') or []):
        return True
    if len(text) > 96:
        return False
    tokens = text.split()
    if len(tokens) > 8:
        return False
    if '.' in text and len(tokens) > 3:
        return False
    return True


def _normalize_for_evidence_match(text: str) -> str:
    """Collapse + HTML entity unescape + light markup unescape for evidence matching."""
    raw = collapse(str(text or ''))
    if not raw:
        return ''
    return collapse(html.unescape(_unescape_source_markup(raw))).casefold()


def _fuzzy_surface_in_quote(surface: str, quote: str, *, threshold: float = 0.92) -> bool:
    """True when a near-equal window of quote matches surface (typos / soft markup)."""
    if not surface or not quote or len(surface) < 12:
        return False
    window = len(surface)
    if window > len(quote):
        return SequenceMatcher(None, surface, quote).ratio() >= threshold
    step = max(1, window // 8)
    best = 0.0
    for i in range(0, len(quote) - window + 1, step):
        best = max(best, SequenceMatcher(None, surface, quote[i:i + window]).ratio())
        if best >= threshold:
            return True
    # Final unaligned compare for short leftover tails.
    return SequenceMatcher(None, surface, quote[-window:]).ratio() >= threshold


def _object_verbatim_in_evidence(surface: str, evidence_quote: str) -> bool:
    """Exact → HTML/markup normalize → light fuzzy. No LLM here."""
    text = collapse(surface or '')
    quote = collapse(evidence_quote or '')
    if not text or not quote:
        return False
    # 1) Exact (casefold substring)
    if text.casefold() in quote.casefold():
        return True
    # 2) HTML entity / markup normalize (e.g. &amp; ↔ &)
    norm_text = _normalize_for_evidence_match(text)
    norm_quote = _normalize_for_evidence_match(quote)
    if norm_text and norm_text in norm_quote:
        return True
    # 3) Fuzzy window match on normalized strings
    return _fuzzy_surface_in_quote(norm_text, norm_quote)


def _infer_unresolved_literal_kind(
    surface: str,
    *,
    predicate: str,
    settings: dict[str, Any],
) -> str:
    """Map unresolved object to a policy literal kind, or '' if entity-shaped/unknown."""
    text = collapse(surface or '')
    if not text:
        return ''
    if _match_identifier_kind(text, settings.get('identifier_patterns') or []):
        return ''
    primary = _infer_recovered_primary(
        text, predicate=predicate, role='object', settings=settings,
    )
    if primary and _surface_is_atomic_endpoint(text, settings):
        return ''
    return 'deliverable_prose'


def _subject_primary_for_literal_contract(
    subject: str,
    *,
    subj_id: str | None,
    index: dict[str, Any],
    by_surface: dict[str, dict],
    policy: Policy,
) -> str:
    """Resolved subject entity primary/label for YAML allowed_subject_primaries checks."""
    by_id = index.get('by_id') if isinstance(index.get('by_id'), dict) else {}
    ent = None
    if subj_id and subj_id in by_id:
        ent = by_id.get(subj_id)
    if ent is None:
        hit = by_surface.get(collapse(subject or '').casefold())
        if isinstance(hit, dict):
            ent = hit
    if not isinstance(ent, dict):
        return ''
    label = collapse(str(ent.get('label') or ''))
    if label:
        return label
    meta = ent.get('metadata') if isinstance(ent.get('metadata'), dict) else {}
    return collapse(str(meta.get('primary_type') or meta.get('type') or ''))


def _apply_object_literal(
    row: dict,
    *,
    obj: str,
    category: str,
    typing_reason: str,
    literal_kind: str = '',
    semantic_validated: bool = False,
) -> None:
    """Mark row object as a literal without minting an entity node."""
    row['object_id'] = None
    row['object_ref_kind'] = 'literal'
    row['object_value_kind'] = 'literal'
    row['object_literal'] = obj
    row['relation_category'] = category
    meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
    meta = dict(meta)
    meta['typing_reason'] = typing_reason
    if literal_kind:
        meta['literal_kind'] = literal_kind
    if semantic_validated:
        meta['semantic_literal_validated'] = True
    else:
        meta.pop('semantic_literal_validated', None)
    row['metadata'] = meta


def recover_missing_endpoints(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    document_text: str = '',
    source_file: str | None = None,
) -> int:
    """Materialize missing endpoints / contract-backed literals from evidence + YAML.

    Entity mint only when atomic, source-supported, and uniquely typed.
    Semantic literals require predicate + literal-kind + subject-role contract.
    Ambiguous objects stay unresolved for quarantine.
    """
    settings = _endpoint_recovery_settings(policy)
    if not settings.get('enabled'):
        return 0
    # Prefix aliasing is intentionally unsupported in this pass.
    if settings.get('allow_prefix_alias'):
        pass

    # Drop previously minted property/literal fakes before resolving again.
    _scrub_invalid_recovered_entities(
        bundle, policy, document_text=document_text,
    )

    index = build_endpoint_resolution_index(bundle.entities, policy=policy)
    by_surface = index['by_surface']
    attr_preds = {p.casefold() for p in attribute_predicate_hints(policy)}
    literal_contracts = settings.get('predicate_literal_object') or {}
    created = 0
    pending: dict[str, dict[str, Any]] = {}
    literal_fixed = 0

    def _consider(
        surface: str,
        *,
        predicate: str,
        role: str,
        evidence_quote: str,
        aliases: list[str] | None = None,
        allow_document_verbatim: bool = True,
    ) -> None:
        nonlocal created
        text = collapse(surface or '')
        if not text:
            return
        key = text.casefold()
        if key in by_surface or key in pending:
            return
        if settings.get('require_verbatim_in_evidence', True):
            quote = (evidence_quote or '').casefold()
            doc = (document_text or '').casefold()
            in_quote = text.casefold() in quote or _object_verbatim_in_evidence(text, evidence_quote)
            in_doc = bool(allow_document_verbatim) and (
                text.casefold() in doc or _object_verbatim_in_evidence(text, document_text)
            )
            if not in_quote and not in_doc:
                return
        validated = _validate_recovered_entity_proposal(
            text,
            predicate=predicate,
            role=role,
            policy=policy,
            evidence_quote=evidence_quote,
            document_text=document_text,
            attribute_object=(role.casefold() == 'object' and (
                predicate.casefold() in attr_preds
                or _is_generic_attribute_predicate(predicate, policy=policy)
                or _looks_like_technical_attribute_claim(
                    predicate,
                    text,
                    evidence_quote=evidence_quote,
                    document_text=document_text,
                    policy=policy,
                )
            )),
        )
        if validated is None:
            return
        primary, subtype = validated
        eid = f'recovered_{hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]}'
        ident_values = [text]
        for alias in aliases or []:
            alias_text = collapse(str(alias or ''))
            if alias_text and alias_text.casefold() != key:
                ident_values.append(alias_text)
        meta_out: dict[str, Any] = {
            'canonical_id': eid,
            'identifiers': ident_values,
            'extraction_method': 'endpoint_recovery',
            'source_file': source_file,
            'typing_reason': 'yaml_endpoint_recovery',
        }
        if subtype:
            meta_out['subtype'] = subtype
        entity = {
            'id': eid,
            'text': text,
            'label': primary,
            'confidence': 0.7,
            'metadata': meta_out,
        }
        pending[key] = entity

    def _try_object_literal(
        row: dict,
        *,
        subject: str,
        obj: str,
        predicate: str,
        quote: str,
        subj_id: str | None,
        is_attr: bool,
    ) -> bool:
        nonlocal literal_fixed
        if not obj:
            return False
        if settings.get('require_verbatim_in_evidence', True) and not _object_verbatim_in_evidence(
            obj, quote,
        ):
            # Fall back to document only for entity mint; literals require evidence quote.
            return False
        if is_attr or (
            predicate.casefold() in attr_preds
            or _is_generic_attribute_predicate(predicate, policy=policy)
            or _looks_like_technical_attribute_claim(
                predicate,
                obj,
                evidence_quote=quote,
                document_text=document_text,
                policy=policy,
            )
        ):
            _apply_object_literal(
                row,
                obj=obj,
                category='attribute',
                typing_reason='evidence_literal_classification',
            )
            literal_fixed += 1
            return True
        contract = literal_contracts.get(predicate.casefold()) if isinstance(literal_contracts, dict) else None
        if not isinstance(contract, dict):
            return False
        kind = _infer_unresolved_literal_kind(obj, predicate=predicate, settings=settings)
        allowed_kinds = {
            collapse(str(k or '')).casefold()
            for k in (contract.get('allowed_literal_kinds') or [])
        }
        if not kind or kind.casefold() not in allowed_kinds:
            return False
        subject_primary = _subject_primary_for_literal_contract(
            subject,
            subj_id=subj_id,
            index=index,
            by_surface=by_surface,
            policy=policy,
        )
        allowed_subjects = {
            collapse(str(s or '')).casefold()
            for s in (contract.get('allowed_subject_primaries') or [])
        }
        if allowed_subjects and subject_primary.casefold() not in allowed_subjects:
            return False
        category = collapse(str(contract.get('category') or 'semantic')) or 'semantic'
        _apply_object_literal(
            row,
            obj=obj,
            category=category,
            typing_reason='evidence_semantic_literal',
            literal_kind=kind,
            semantic_validated=True,
        )
        literal_fixed += 1
        return True

    for rel in bundle.relations:
        subject = collapse(str(rel.get('subject') or ''))
        obj = collapse(str(rel.get('object') or ''))
        predicate = collapse(str(rel.get('predicate') or ''))
        quote = _evidence_quote_for_row(rel)
        subj_id, subj_kind = _resolve_endpoint_ref(
            subject, by_surface=by_surface, index=index, policy=policy,
        )
        obj_id, obj_kind = _resolve_endpoint_ref(
            obj, by_surface=by_surface, index=index, policy=policy,
        )
        if subj_kind != 'entity':
            _consider(subject, predicate=predicate, role='subject', evidence_quote=quote)
        if obj_kind != 'entity':
            if not _try_object_literal(
                rel,
                subject=subject,
                obj=obj,
                predicate=predicate,
                quote=quote,
                subj_id=subj_id if subj_kind == 'entity' else None,
                is_attr=False,
            ):
                _consider(obj, predicate=predicate, role='object', evidence_quote=quote)

    for trip in bundle.triplets:
        subject = collapse(str(trip.get('subject') or ''))
        obj = collapse(str(trip.get('object') or ''))
        predicate = collapse(str(trip.get('predicate') or ''))
        quote = _evidence_quote_for_row(trip)
        evidence_quote = quote
        is_attr = (
            predicate.casefold() in attr_preds
            or _is_generic_attribute_predicate(predicate, policy=policy)
            or collapse(str(trip.get('relation_category') or '')).casefold() == 'attribute'
            or (
                collapse(str(trip.get('object_value_kind') or '')).casefold() == 'literal'
                and not (
                    isinstance(trip.get('metadata'), dict)
                    and trip['metadata'].get('semantic_literal_validated')
                )
            )
            or _looks_like_technical_attribute_claim(
                predicate,
                obj,
                evidence_quote=evidence_quote,
                document_text=document_text,
                policy=policy,
            )
        )
        trip_meta = trip.get('metadata') if isinstance(trip.get('metadata'), dict) else {}
        subj_id, subj_kind = _resolve_endpoint_via_provenance(
            subject,
            trip_meta,
            by_surface=by_surface,
            index=index,
            policy=policy,
        )
        if subj_kind != 'entity':
            section = _section_context_for_row(trip)
            owner = collapse(str(section.get('source_text') or ''))
            if _subject_grounded_in_provenance(subject, trip_meta, policy=policy):
                # Prefer verbatim owner-cell surface from section_context when the
                # LLM subject was a split-row concat not contiguous in the source.
                if owner and (
                    _object_verbatim_in_evidence(owner, document_text)
                    or _object_verbatim_in_evidence(owner, quote)
                ):
                    _consider(
                        owner,
                        predicate=predicate,
                        role='subject',
                        evidence_quote=owner,
                        aliases=[subject],
                        allow_document_verbatim=True,
                    )
                _consider(
                    subject,
                    predicate=predicate,
                    role='subject',
                    evidence_quote=quote or owner,
                    aliases=[owner] if owner else None,
                )
            else:
                _consider(subject, predicate=predicate, role='subject', evidence_quote=quote)
        obj_id, obj_kind = _resolve_endpoint_ref(
            obj,
            by_surface=by_surface,
            attribute_mode=is_attr,
            index=index,
            policy=policy,
        )
        if obj_kind != 'entity':
            applied = _try_object_literal(
                trip,
                subject=subject,
                obj=obj,
                predicate=predicate,
                quote=quote,
                subj_id=subj_id if subj_kind == 'entity' else None,
                is_attr=is_attr,
            )
            if not applied and not is_attr:
                _consider(obj, predicate=predicate, role='object', evidence_quote=quote)

    for entity in pending.values():
        bundle.entities.append(entity)
        created += 1
    if created or literal_fixed:
        _diag_record(
            'endpoint_recovery',
            created=created,
            literal_fixed=literal_fixed,
            surfaces=list(pending.keys())[:20],
        )
    return created


def validate_extraction_graph(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    source_file: str | None = None,
    document_text: str = '',
) -> list[str]:
    """Post-extraction consistency checks; annotate endpoints; report warnings."""
    warnings: list[str] = []
    entities = bundle.entities
    index = build_endpoint_resolution_index(entities, policy=policy)
    by_id = index['by_id']
    by_surface = index['by_surface']
    attr_preds = {p.casefold() for p in attribute_predicate_hints(policy)}
    hier_preds = hierarchy_predicate_set(policy)
    allowed_pairs = allowed_parent_child_pairs(policy)
    subtype_gates = predicate_subtype_gates(policy)
    hierarchy = policy.data.get('hierarchy') if isinstance(policy.data, dict) else {}
    if not isinstance(hierarchy, dict):
        hierarchy = {}
    forbid_self = bool(hierarchy.get('forbid_self_parent', True))
    prevent_cycles = bool(hierarchy.get('prevent_cycles', True))
    enforce_allowed = bool(hierarchy.get('enforce_allowed_parent_child', True))

    parent_child_edges: list[tuple[str, str]] = []
    reverse_seen: dict[tuple[str, str], str] = {}

    def _entity_subtype(entity_id: str | None) -> str:
        if not entity_id:
            return ''
        meta = (by_id.get(entity_id) or {}).get('metadata')
        if not isinstance(meta, dict):
            return ''
        return collapse(str(meta.get('subtype') or ''))

    def _entity_label(entity_id: str | None) -> str:
        if not entity_id:
            return ''
        return collapse(str((by_id.get(entity_id) or {}).get('label') or '')).casefold()

    def _check_subtype_gate(predicate: str, parent_id: str | None, child_id: str | None) -> None:
        gate = subtype_gates.get(predicate.casefold())
        if not gate or not parent_id or not child_id:
            return
        reason = _subtype_gate_violation(
            gate,
            parent_subtype=_entity_subtype(parent_id),
            child_subtype=_entity_subtype(child_id),
        )
        if reason:
            warnings.append(
                f'hierarchy subtype gate for {predicate!r}: {reason}'
            )

    def _record_hierarchy(predicate: str, subj_id: str, obj_id: str, *, subject: str, obj: str) -> None:
        parent_id, child_id, orientation = canonicalize_hierarchy_edge(
            subj_id, obj_id, predicate, policy=policy,
        )
        parent_child_edges.append((parent_id, child_id))
        key = (parent_id, child_id)
        rev = (child_id, parent_id)
        if rev in reverse_seen and reverse_seen[rev] != predicate.casefold():
            # Inverse pair of the same containment is not a contradiction.
            if orientation == 'inverted_for_hierarchy' or reverse_seen[rev] in hierarchy_child_as_subject_predicates(policy):
                pass
            else:
                warnings.append(
                    f'contradictory hierarchy links between {subject!r} and {obj!r} '
                    f'({reverse_seen[rev]} vs {predicate})'
                )
        reverse_seen[key] = predicate.casefold()
        if enforce_allowed and allowed_pairs:
            parent_label = _entity_label(parent_id)
            child_label = _entity_label(child_id)
            if parent_label and child_label and (parent_label, child_label) not in allowed_pairs:
                warnings.append(
                    f'invalid parent-child types for {predicate!r}: {parent_label}->{child_label}'
                )
        _check_subtype_gate(predicate, parent_id, child_id)

    for rel in bundle.relations:
        subject = collapse(str(rel.get('subject') or ''))
        obj = collapse(str(rel.get('object') or ''))
        predicate = collapse(str(rel.get('predicate') or ''))
        rel_meta = rel.get('metadata') if isinstance(rel.get('metadata'), dict) else {}
        semantic_lit = bool(rel_meta.get('semantic_literal_validated'))
        subj_id, subj_kind = _resolve_endpoint_via_provenance(
            subject,
            rel_meta,
            by_surface=by_surface,
            index=index,
            policy=policy,
        )
        if semantic_lit and collapse(str(rel.get('object_value_kind') or '')).casefold() == 'literal':
            obj_id, obj_kind = None, 'literal'
            rel['object_literal'] = obj
            rel['relation_category'] = (
                collapse(str(rel.get('relation_category') or ''))
                or classify_relation_category(predicate, policy=policy)
            )
        else:
            obj_id, obj_kind = _resolve_endpoint_ref(
                obj, by_surface=by_surface, index=index, policy=policy,
            )
            rel['relation_category'] = classify_relation_category(predicate, policy=policy)
        rel['subject_id'] = subj_id
        rel['object_id'] = obj_id
        rel['subject_ref_kind'] = subj_kind
        rel['object_ref_kind'] = obj_kind
        if semantic_lit:
            rel['object_value_kind'] = 'literal'
        if subj_kind != 'entity' and not _subject_grounded_in_provenance(
            subject, rel_meta, policy=policy,
        ):
            warnings.append(f'missing entity node for relation subject: {subject!r}')
        if obj_kind != 'entity' and not semantic_lit:
            warnings.append(f'missing entity node for relation object: {obj!r}')
        if forbid_self and (
            (subj_id and obj_id and subj_id == obj_id)
            or subject.casefold() == obj.casefold()
        ):
            warnings.append(f'self-parent relation forbidden: {subject!r} -[{predicate}]-> {obj!r}')
        if (
            not semantic_lit
            and classify_relation_category(predicate, policy=policy) in {
                'engineering_hierarchy', 'document_hierarchy',
            }
            and predicate.casefold() in hier_preds
            and subj_id and obj_id
        ):
            _record_hierarchy(predicate, subj_id, obj_id, subject=subject, obj=obj)

    for trip in bundle.triplets:
        subject = collapse(str(trip.get('subject') or ''))
        obj = collapse(str(trip.get('object') or ''))
        predicate = collapse(str(trip.get('predicate') or ''))
        trip_meta = trip.get('metadata') if isinstance(trip.get('metadata'), dict) else {}
        evidence_quote = ''
        if isinstance(trip_meta.get('evidence'), dict):
            evidence_quote = str(trip_meta['evidence'].get('source_text') or '')
        semantic_lit = bool(trip_meta.get('semantic_literal_validated'))
        is_attr = (not semantic_lit) and (
            predicate.casefold() in attr_preds
            or _is_generic_attribute_predicate(predicate, policy=policy)
            or collapse(str(trip.get('relation_category') or '')).casefold() == 'attribute'
            or (
                collapse(str(trip.get('object_value_kind') or '')).casefold() == 'literal'
                and not semantic_lit
            )
            or _looks_like_technical_attribute_claim(
                predicate,
                obj,
                evidence_quote=evidence_quote,
                document_text=document_text,
                policy=policy,
            )
        )
        subj_id, subj_kind = _resolve_endpoint_via_provenance(
            subject,
            trip_meta,
            by_surface=by_surface,
            index=index,
            policy=policy,
        )
        if semantic_lit:
            obj_id, obj_kind = None, 'literal'
            trip['object_literal'] = obj
            trip['object_value_kind'] = 'literal'
            trip['relation_category'] = (
                collapse(str(trip.get('relation_category') or '')) or 'semantic'
            )
        else:
            obj_id, obj_kind = _resolve_endpoint_ref(
                obj, by_surface=by_surface, attribute_mode=is_attr, index=index, policy=policy,
            )
            trip['object_value_kind'] = 'literal' if is_attr and obj_kind != 'entity' else obj_kind
            trip['relation_category'] = 'attribute' if is_attr else classify_relation_category(
                predicate, policy=policy,
            )
        trip['subject_id'] = subj_id
        trip['object_id'] = obj_id
        trip['subject_ref_kind'] = subj_kind
        trip['object_ref_kind'] = obj_kind
        if subj_kind != 'entity' and not _subject_grounded_in_provenance(
            subject, trip_meta, policy=policy,
        ):
            warnings.append(f'missing entity node for triplet subject: {subject!r}')
        if not is_attr and not semantic_lit and obj_kind != 'entity':
            warnings.append(f'missing entity node for triplet object: {obj!r}')
        if (
            not is_attr
            and not semantic_lit
            and classify_relation_category(predicate, policy=policy) in {
                'engineering_hierarchy', 'document_hierarchy',
            }
            and predicate.casefold() in hier_preds
            and subj_id and obj_id
        ):
            _record_hierarchy(predicate, subj_id, obj_id, subject=subject, obj=obj)

    if prevent_cycles and parent_child_edges:
        cyclic = _hierarchy_has_cycle(parent_child_edges)
        if cyclic:
            warnings.append(f'hierarchy cycle involving entity ids: {", ".join(cyclic)}')

    # Possible cross-document duplicates from prior output files (warn only).
    prior = load_prior_graph_entities(exclude_source=source_file)
    if prior:
        prior_by_surface: dict[str, list[str]] = {}
        for entity in prior:
            surface = collapse(str(entity.get('text') or '')).casefold()
            src = collapse(str(entity.get('_prior_source') or ''))
            if surface:
                prior_by_surface.setdefault(surface, []).append(src)
            meta = entity.get('metadata') if isinstance(entity.get('metadata'), dict) else {}
            for ident in _identifier_values(meta):
                prior_by_surface.setdefault(ident, []).append(src)
        for entity in entities:
            surface = collapse(str(entity.get('text') or '')).casefold()
            meta = entity.get('metadata') if isinstance(entity.get('metadata'), dict) else {}
            keys = {surface} | _identifier_values(meta)
            for key in keys:
                if not key:
                    continue
                hits = [s for s in prior_by_surface.get(key, []) if s]
                if hits:
                    warnings.append(
                        f'possible cross-document duplicate for {entity.get("text")!r} '
                        f'also seen in {sorted(set(hits))[:3]}'
                    )
                    break

    return warnings


def _edge_reason_codes(
    *,
    subj_kind: str,
    obj_kind: str,
    is_attr: bool,
    meta: dict,
    category: str,
    hierarchy_ok: bool,
    self_link: bool,
    literal_object: bool = False,
    merged_identity_self_loop: bool = False,
    claim_source_backed: bool = True,
    evidence_backed_subject: bool = False,
) -> list[str]:
    reasons: list[str] = []
    # Subject grounded in evidence/section_context (exact/fuzzy/tag) is not unresolved.
    if subj_kind != 'entity' and not evidence_backed_subject:
        reasons.append('unresolved_subject')
    if not is_attr and not literal_object and obj_kind != 'entity':
        reasons.append('unresolved_object')
    status = collapse(str(meta.get('evidence_status') or 'missing')).casefold()
    if status == 'missing':
        reasons.append('evidence_missing')
    elif status == 'unverified':
        reasons.append('evidence_unverified')
    if not meta.get('graph_eligible'):
        reasons.append('not_graph_eligible')
    if not claim_source_backed:
        reasons.append('claim_not_source_backed')
    if merged_identity_self_loop:
        reasons.append('merged_identity_self_loop')
    elif self_link:
        reasons.append('self_parent')
    if category in {'engineering_hierarchy', 'document_hierarchy'} and not hierarchy_ok:
        reasons.append('hierarchy_invalid')
    if meta.get('predicate_contract_type_violation'):
        reasons.append('predicate_contract_type_violation')
    return reasons or ['ok']


def build_confirmed_graph(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    source_file: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split relations/triplets into confirmed graph edges vs quarantine review rows.

    Does not delete extracted facts. Confirmed edges require resolved endpoints
    (literals allowed for attributes), verified non-weak evidence, and valid
    hierarchy constraints when the predicate is hierarchical.
    """
    hierarchy = policy.data.get('hierarchy') if isinstance(policy.data, dict) else {}
    if not isinstance(hierarchy, dict):
        hierarchy = {}
    quarantine_invalid = bool(hierarchy.get('quarantine_invalid_hierarchy', True))
    forbid_self = bool(hierarchy.get('forbid_self_parent', True))
    enforce_allowed = bool(hierarchy.get('enforce_allowed_parent_child', True))
    allowed_pairs = allowed_parent_child_pairs(policy)
    index = build_endpoint_resolution_index(bundle.entities, policy=policy)
    by_id = index['by_id']

    nodes = []
    for entity in bundle.entities:
        eid = collapse(str(entity.get('id') or (entity.get('metadata') or {}).get('canonical_id') or ''))
        if not eid:
            continue
        nodes.append({
            'id': eid,
            'text': collapse(str(entity.get('text') or '')),
            'label': collapse(str(entity.get('label') or '')),
            'subtype': collapse(str((entity.get('metadata') or {}).get('subtype') or '')),
            'identifiers': list((entity.get('metadata') or {}).get('identifiers') or []),
        })

    confirmed_edges: list[dict[str, Any]] = []
    quarantine_relations: list[dict[str, Any]] = []
    quarantine_triplets: list[dict[str, Any]] = []
    seen_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    hierarchy_edges: list[tuple[str, str]] = []

    def _labels(eid: str | None) -> str:
        if not eid:
            return ''
        return collapse(str((by_id.get(eid) or {}).get('label') or '')).casefold()

    def _hierarchy_valid(pred: str, subj_id: str | None, obj_id: str | None) -> bool:
        if not subj_id or not obj_id:
            return False
        parent_id, child_id, _orient = canonicalize_hierarchy_edge(
            subj_id, obj_id, pred, policy=policy,
        )
        if forbid_self and parent_id == child_id:
            return False
        if enforce_allowed and allowed_pairs:
            pl, cl = _labels(parent_id), _labels(child_id)
            if pl and cl and (pl, cl) not in allowed_pairs:
                return False
        gate = predicate_subtype_gates(policy).get(pred.casefold())
        if gate:
            parent_meta = (by_id.get(parent_id) or {}).get('metadata') or {}
            child_meta = (by_id.get(child_id) or {}).get('metadata') or {}
            if _subtype_gate_violation(
                gate,
                parent_subtype=collapse(str(parent_meta.get('subtype') or '')),
                child_subtype=collapse(str(child_meta.get('subtype') or '')),
            ):
                return False
        hierarchy_edges.append((parent_id, child_id))
        return True

    def _handle_row(row: dict, *, kind: str) -> None:
        subject = collapse(str(row.get('subject') or ''))
        obj = collapse(str(row.get('object') or ''))
        predicate = collapse(str(row.get('predicate') or ''))
        meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
        evidence_quote = ''
        if isinstance(row.get('evidence'), dict):
            evidence_quote = str(row['evidence'].get('source_text') or '')
        elif isinstance(meta.get('evidence'), dict):
            evidence_quote = str(meta['evidence'].get('source_text') or '')
        # Semantic literals are accepted only when recovery already validated them.
        semantic_lit = bool(meta.get('semantic_literal_validated')) and (
            collapse(str(row.get('object_ref_kind') or '')).casefold() == 'literal'
            or collapse(str(row.get('object_value_kind') or '')).casefold() == 'literal'
        )
        is_attr = (not semantic_lit) and kind == 'triplet' and (
            collapse(str(row.get('relation_category') or '')).casefold() == 'attribute'
            or collapse(str(row.get('object_value_kind') or '')).casefold() == 'literal'
            or _is_generic_attribute_predicate(predicate, policy=policy)
            or _looks_like_technical_attribute_claim(
                predicate,
                obj,
                evidence_quote=evidence_quote,
                policy=policy,
            )
        )
        category = collapse(str(row.get('relation_category') or '')) or classify_relation_category(
            predicate, policy=policy,
        )
        if is_attr:
            category = 'attribute'
        elif semantic_lit:
            category = collapse(str(row.get('relation_category') or '')) or 'semantic'
        literal_object = bool(is_attr or semantic_lit)
        subj_id = row.get('subject_id')
        obj_id = row.get('object_id')
        subj_kind = collapse(str(row.get('subject_ref_kind') or ''))
        obj_kind = collapse(str(row.get('object_ref_kind') or ''))
        if semantic_lit:
            obj_kind = 'literal'
            obj_id = None
        is_hierarchy = category in {'engineering_hierarchy', 'document_hierarchy'}
        # Hierarchy self-parent only (YAML forbid_self_parent). Semantic self-loops
        # are not auto-removed here — existing predicate/YAML rules apply elsewhere.
        same_endpoint = bool(
            (subj_id and obj_id and subj_id == obj_id)
            or subject.casefold() == obj.casefold()
        )
        self_link = bool(forbid_self and is_hierarchy and same_endpoint)
        # Distinct surfaces collapsed to one id after identity merge.
        merged_identity_self_loop = bool(
            self_link
            and subj_id
            and obj_id
            and subj_id == obj_id
            and subject.casefold() != obj.casefold()
        )
        hierarchy_ok = True
        if is_hierarchy:
            hierarchy_ok = _hierarchy_valid(predicate, subj_id, obj_id)
        # Re-resolve subject via evidence/section_context when still open.
        if subj_kind != 'entity' or not subj_id:
            eid2, kind2 = _resolve_endpoint_via_provenance(
                subject,
                meta,
                by_surface=index.get('by_surface') or {},
                index=index,
                policy=policy,
            )
            if kind2 == 'entity' and eid2:
                subj_id = eid2
                subj_kind = kind2
                row['subject_id'] = eid2
                row['subject_ref_kind'] = 'entity'
        # Object text present in evidence quote → do not flag unresolved_object.
        # Does not mint a graph literal or relax eligibility (needs attr/semantic contract).
        evidence_backed_object = bool(
            not literal_object
            and not obj_id
            and obj
            and _object_verbatim_in_evidence(obj, evidence_quote)
        )
        evidence_backed_subject = bool(
            (subj_kind != 'entity' or not subj_id)
            and _subject_grounded_in_provenance(subject, meta, policy=policy)
        )
        # Re-check source backing at confirm time (subject + predicate meaning +
        # object/value + direction). Co-mention alone must not confirm an edge.
        # Attributes may ground the owner via section_context outside the value quote.
        claim_source_backed = True
        if evidence_quote and subject and obj and predicate:
            if is_attr or semantic_lit:
                if _is_incomplete_attribute_action_value(
                    obj, evidence_quote, predicate, policy=policy,
                ):
                    claim_source_backed = False
                elif not _quote_value_supported(evidence_quote, obj, policy=policy):
                    claim_source_backed = False
                else:
                    owner_in_quote = _endpoint_supported_in_quote(
                        subject, evidence_quote.casefold(), policy=policy,
                    )
                    owner_via_section = bool(meta.get('section_context'))
                    claim_source_backed = bool(owner_in_quote or owner_via_section)
            else:
                claim_source_backed = _claim_supported_by_quote(
                    evidence_quote,
                    subject,
                    obj,
                    attribute_mode=False,
                    predicate=predicate,
                    policy=policy,
                )
        elif meta.get('graph_eligible') is True:
            claim_source_backed = False
        reasons = _edge_reason_codes(
            subj_kind=subj_kind or ('entity' if subj_id else 'unresolved'),
            obj_kind=obj_kind or (
                'literal' if literal_object else ('entity' if obj_id else 'unresolved')
            ),
            is_attr=is_attr,
            meta=meta,
            category=category,
            hierarchy_ok=hierarchy_ok,
            self_link=self_link,
            literal_object=literal_object or evidence_backed_object,
            merged_identity_self_loop=merged_identity_self_loop,
            claim_source_backed=claim_source_backed,
            evidence_backed_subject=evidence_backed_subject,
        )
        eligible = (
            meta.get('graph_eligible') is True
            and collapse(str(meta.get('evidence_status') or '')).casefold() == 'verified'
            and (subj_kind == 'entity' or bool(subj_id))
            and (literal_object or obj_kind == 'entity' or bool(obj_id))
            and not self_link
            and (hierarchy_ok or not is_hierarchy)
            and claim_source_backed
        )
        if quarantine_invalid and category in {'engineering_hierarchy', 'document_hierarchy'} and not hierarchy_ok:
            eligible = False
            if 'hierarchy_invalid' not in reasons:
                reasons.append('hierarchy_invalid')

        edge = {
            'kind': kind,
            'subject': subject,
            'predicate': predicate,
            'object': obj,
            'subject_id': subj_id,
            'object_id': None if literal_object else obj_id,
            'object_literal': obj if literal_object else None,
            'relation_category': category,
            'evidence_status': meta.get('evidence_status'),
            'graph_eligible_flag': meta.get('graph_eligible'),
            'source_file': source_file,
            'evidence': meta.get('evidence'),
            'section_context': meta.get('section_context'),
        }
        if eligible:
            dedupe_key = (
                str(subj_id or subject).casefold(),
                predicate.casefold(),
                str((obj if literal_object else obj_id) or '').casefold(),
            )
            if dedupe_key in seen_edges:
                # Keep extra verified spans on the first confirmed edge (no quarantine).
                primary = seen_edges[dedupe_key]
                extra = meta.get('evidence')
                if isinstance(extra, dict) and extra:
                    bucket = primary.setdefault('supporting_evidence', [])
                    fingerprints = set()
                    for existing in ([primary.get('evidence')] if isinstance(primary.get('evidence'), dict) else []):
                        fingerprints.add(json.dumps(existing, sort_keys=True, default=str))
                    for existing in bucket:
                        if isinstance(existing, dict):
                            fingerprints.add(json.dumps(existing, sort_keys=True, default=str))
                    fp = json.dumps(extra, sort_keys=True, default=str)
                    if fp not in fingerprints:
                        bucket.append(extra)
                return
            if category in {'engineering_hierarchy', 'document_hierarchy'} and subj_id and obj_id:
                parent_id, child_id, orientation = canonicalize_hierarchy_edge(
                    subj_id, obj_id, predicate, policy=policy,
                )
                edge['hierarchy_parent_id'] = parent_id
                edge['hierarchy_child_id'] = child_id
                edge['hierarchy_orientation'] = orientation
            seen_edges[dedupe_key] = edge
            confirmed_edges.append(edge)
            return

        quarantine_row = dict(edge)
        quarantine_row['reason_codes'] = [r for r in reasons if r != 'ok'] or ['review']
        suggested = None
        if 'merged_identity_self_loop' in quarantine_row['reason_codes']:
            suggested = 'drop_or_merge_duplicate_hierarchy_endpoint'
        elif 'claim_not_source_backed' in quarantine_row['reason_codes']:
            suggested = 'recover_or_review_source_evidence'
        elif 'unresolved_subject' in quarantine_row['reason_codes'] or 'unresolved_object' in quarantine_row['reason_codes']:
            suggested = 'resolve_endpoints_via_identifiers_or_aliases'
        elif 'evidence_missing' in quarantine_row['reason_codes'] or 'evidence_unverified' in quarantine_row['reason_codes']:
            suggested = 'recover_or_review_source_evidence'
        elif 'hierarchy_invalid' in quarantine_row['reason_codes']:
            suggested = 'review_predicate_or_direction_against_ontology'
        elif 'self_parent' in quarantine_row['reason_codes']:
            suggested = 'review_predicate_or_direction_against_ontology'
        quarantine_row['suggested_repair'] = suggested
        quarantine_row['recovery_attempted'] = bool(meta.get('evidence_recovery'))
        if 'merged_identity_self_loop' in quarantine_row['reason_codes']:
            quarantine_row['audit_reason'] = 'merged_identity_self_loop'
            quarantine_row['preserved_subject'] = subject
            quarantine_row['preserved_object'] = obj
            quarantine_row['preserved_subject_id'] = subj_id
            quarantine_row['preserved_object_id'] = obj_id
        if 'claim_not_source_backed' in quarantine_row['reason_codes']:
            quarantine_row['audit_reason'] = quarantine_row.get('audit_reason') or 'claim_not_source_backed'
            quarantine_row['preserved_evidence'] = meta.get('evidence')
            quarantine_row['preserved_section_context'] = meta.get('section_context')
        (quarantine_triplets if kind == 'triplet' else quarantine_relations).append(quarantine_row)

    for rel in bundle.relations:
        _handle_row(rel, kind='relation')
    for trip in bundle.triplets:
        _handle_row(trip, kind='triplet')

    # Drop confirmed hierarchy edges that participate in cycles (keep in quarantine).
    cyclic_nodes = set(_hierarchy_has_cycle(hierarchy_edges)) if hierarchy_edges else set()
    if cyclic_nodes:
        kept: list[dict[str, Any]] = []
        for edge in confirmed_edges:
            parent = edge.get('hierarchy_parent_id')
            child = edge.get('hierarchy_child_id')
            if parent in cyclic_nodes and child in cyclic_nodes:
                q = dict(edge)
                q['reason_codes'] = ['hierarchy_cycle']
                q['suggested_repair'] = 'break_cycle_using_source_evidence'
                quarantine_relations.append(q)
            else:
                kept.append(edge)
        confirmed_edges = kept

    reason_counts: dict[str, int] = {}
    for row in quarantine_relations + quarantine_triplets:
        for code in row.get('reason_codes') or []:
            reason_counts[str(code)] = reason_counts.get(str(code), 0) + 1

    graph = {
        'source_file': source_file,
        'node_count': len(nodes),
        'edge_count': len(confirmed_edges),
        'nodes': nodes,
        'edges': confirmed_edges,
    }
    quarantine = {
        'relation_count': len(quarantine_relations),
        'triplet_count': len(quarantine_triplets),
        'reason_counts': reason_counts,
        'relations': quarantine_relations,
        'triplets': quarantine_triplets,
    }
    return graph, quarantine


def _quarantine_llm_settings(policy: Policy | None = None) -> dict[str, Any]:
    pol = policy if policy is not None else load_policy()
    data = pol.data if isinstance(pol.data, dict) else {}
    cfg = data.get('quarantine_llm_resolver') if isinstance(data.get('quarantine_llm_resolver'), dict) else {}

    def _codes(key: str, default: list[str]) -> set[str]:
        raw = cfg.get(key) if isinstance(cfg.get(key), list) else default
        return {collapse(str(x)).casefold() for x in raw if collapse(str(x))}

    try:
        max_n = int(cfg.get('max_candidates_per_file', 24))
    except (TypeError, ValueError):
        max_n = 24
    try:
        batch = int(cfg.get('batch_size', 6))
    except (TypeError, ValueError):
        batch = 6
    try:
        ctx = int(cfg.get('context_chars', 900))
    except (TypeError, ValueError):
        ctx = 900
    return {
        'enabled': bool(cfg.get('enabled', False)),
        'max_candidates_per_file': max(0, min(max_n, 80)),
        'batch_size': max(1, min(batch, 12)),
        'context_chars': max(200, min(ctx, 4000)),
        'eligible': _codes('eligible_reason_codes', [
            'evidence_missing', 'evidence_unverified',
            'unresolved_subject', 'unresolved_object', 'hierarchy_invalid',
        ]),
        'excluded': _codes('excluded_reason_codes', [
            'duplicate_edge', 'self_parent', 'hierarchy_cycle',
        ]),
        'hierarchy_repair_predicates_only': bool(cfg.get('hierarchy_repair_predicates_only', True)),
    }


def _primary_type_names(policy: Policy) -> list[str]:
    data = policy.data if isinstance(policy.data, dict) else {}
    taxonomy = data.get('ner_taxonomy') if isinstance(data.get('ner_taxonomy'), dict) else {}
    names: list[str] = []
    for row in taxonomy.get('entity_types') or []:
        if isinstance(row, dict):
            name = collapse(str(row.get('type') or ''))
        else:
            name = collapse(str(row or ''))
        if name:
            names.append(name)
    return names


def _parse_json_list(text: str) -> list[dict]:
    raw = text or ''
    start = raw.find('[')
    end = raw.rfind(']')
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _context_window(document_text: str, needle: str, *, limit: int) -> str:
    doc = document_text or ''
    key = collapse(needle or '')
    if not doc:
        return ''
    idx = doc.casefold().find(key.casefold()) if key else -1
    if idx < 0:
        return doc[:limit]
    half = max(40, limit // 2)
    lo = max(0, idx - half)
    hi = min(len(doc), idx + max(len(key), 1) + half)
    return doc[lo:hi]


def _default_quarantine_llm(prompt: str) -> str | None:
    try:
        provider = BedrockProvider()
        if not provider.is_available():
            return None
        return provider.generate(prompt, max_tokens=2048, temperature=0.0)
    except Exception:
        return None


def resolve_quarantine_with_llm(
    bundle: ExtractionBundle,
    policy: Policy,
    *,
    document_text: str = '',
    source_file: str | None = None,
    resolver: Callable[..., str | None] | None = None,
) -> int:
    """Propose repairs for selected quarantine rows, then re-validate with YAML.

    Local Exact → HTML-normalize → fuzzy evidence grounding runs first; the 2nd
    LLM is only invoked for rows where that cascade fails (or hierarchy repair
    is needed). LLM cannot relax hierarchy gates or merge distinct identifier
    surfaces. Returns how many source rows were rewritten before the rebuild.
    """
    settings = _quarantine_llm_settings(policy)
    if not settings.get('enabled'):
        return 0
    quarantine = bundle.quarantine if isinstance(bundle.quarantine, dict) else {}
    hier_preds = hierarchy_predicate_set(policy)
    selected: list[dict[str, Any]] = []
    for kind in ('relation', 'triplet'):
        rows = quarantine.get('relations' if kind == 'relation' else 'triplets') or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            codes = {collapse(str(c)).casefold() for c in (row.get('reason_codes') or [])}
            if codes & settings['excluded']:
                continue
            if not (codes & settings['eligible']):
                continue
            # Exact → Normalize → Fuzzy first; 2nd LLM only when local grounding fails
            # or hierarchy/predicate repair is still needed (entity↔entity non-hierarchy).
            evidence = row.get('evidence') if isinstance(row.get('evidence'), dict) else {}
            quote = str(evidence.get('source_text') or '')
            obj = collapse(str(row.get('object') or ''))
            subj = collapse(str(row.get('subject') or ''))
            needs_obj = bool(codes & {'unresolved_object', 'evidence_missing', 'evidence_unverified'})
            needs_subj = bool(codes & {'unresolved_subject', 'evidence_missing', 'evidence_unverified'})
            pred = collapse(str(row.get('predicate') or ''))
            could_hierarchy_repair = bool(codes & {'hierarchy_invalid'}) or (
                bool(row.get('subject_id') and row.get('object_id'))
                and pred.casefold() not in hier_preds
            )
            if quote and (needs_obj or needs_subj) and not could_hierarchy_repair:
                obj_ok = (not needs_obj) or (not obj) or _object_verbatim_in_evidence(obj, quote)
                subj_ok = (not needs_subj) or (not subj) or _object_verbatim_in_evidence(subj, quote)
                if obj_ok and subj_ok:
                    # Local cascade already grounds claim text; keep for Python/YAML only.
                    continue
            item = dict(row)
            item['kind'] = kind
            selected.append(item)
            if len(selected) >= int(settings['max_candidates_per_file']):
                break
        if len(selected) >= int(settings['max_candidates_per_file']):
            break
    if not selected:
        return 0

    ask = resolver if resolver is not None else _default_quarantine_llm
    primaries = _primary_type_names(policy)
    hier_preds = sorted(hierarchy_predicate_set(policy))
    known = [
        f"{collapse(str(e.get('text') or ''))} [{collapse(str(e.get('label') or ''))}]"
        for e in (bundle.entities or [])[:40]
        if isinstance(e, dict) and collapse(str(e.get('text') or ''))
    ]
    ctx_limit = int(settings['context_chars'])
    applied = 0
    confirmed_applied = 0
    batch_size = int(settings['batch_size'])

    def _lookup_source(item: dict) -> dict | None:
        pool = bundle.relations if item.get('kind') == 'relation' else bundle.triplets
        subj = collapse(str(item.get('subject') or '')).casefold()
        pred = collapse(str(item.get('predicate') or '')).casefold()
        obj = collapse(str(item.get('object') or '')).casefold()
        for row in pool:
            if not isinstance(row, dict):
                continue
            if (
                collapse(str(row.get('subject') or '')).casefold() == subj
                and collapse(str(row.get('predicate') or '')).casefold() == pred
                and collapse(str(row.get('object') or '')).casefold() == obj
            ):
                return row
        return None

    for offset in range(0, len(selected), batch_size):
        batch = selected[offset:offset + batch_size]
        payload = []
        for i, row in enumerate(batch):
            quote = ''
            evidence = row.get('evidence') if isinstance(row.get('evidence'), dict) else {}
            quote = collapse(str(evidence.get('source_text') or ''))
            window = _context_window(
                document_text,
                quote or collapse(str(row.get('subject') or '')),
                limit=ctx_limit,
            )
            payload.append({
                'id': i,
                'kind': row.get('kind'),
                'subject': row.get('subject'),
                'predicate': row.get('predicate'),
                'object': row.get('object'),
                'reason_codes': row.get('reason_codes') or [],
                'current_quote': quote[:400],
                'source_window': window,
            })
        prompt = '\n'.join([
            'You repair quarantined knowledge-graph claims. Reply with JSON array only.',
            'Each item: {"id": n, "action": "accept"|"repair"|"reject",',
            ' "subject": "", "predicate": "", "object": "", "evidence_quote": "",',
            ' "entity_type_hint": ""}',
            'Rules:',
            '- evidence_quote must be an exact contiguous substring of source_window.',
            '- Do not invent entities, identifiers, or hierarchy.',
            '- accept keeps subject/predicate/object and only supplies a better verbatim quote.',
            '- repair may change predicate or endpoint text when the window supports it.',
            '- reject when the claim is unsupported.',
            '- entity_type_hint must be one of the allowed primaries or empty.',
            '- Containment predicates may be used only when the window states membership.',
            '- Action/control wording is not containment.',
            f'Allowed primaries: {", ".join(primaries[:40])}',
            f'Hierarchy predicates: {", ".join(hier_preds[:30])}',
            'Known entities:',
            '\n'.join(f'- {line}' for line in known[:30]) or '- (none)',
            'Claims:',
            json.dumps(payload, ensure_ascii=False),
        ])
        try:
            reply = ask(prompt)
        except Exception:
            reply = None
        decisions: dict[int, dict] = {}
        for decision_row in _parse_json_list(str(reply or '')):
            try:
                decisions[int(decision_row.get('id'))] = decision_row
            except (TypeError, ValueError):
                continue
        allowed_types = {name.casefold(): name for name in primaries}
        for i, item in enumerate(batch):
            decision = decisions.get(i)
            if not decision:
                continue
            action = collapse(str(decision.get('action') or '')).casefold()
            if action not in {'accept', 'repair'}:
                continue
            # Keep verbatim newlines so multi-line hierarchy/table quotes still match.
            quote = str(decision.get('evidence_quote') or '').strip()
            if not quote or quote not in (document_text or ''):
                continue
            source = _lookup_source(item)
            if source is None:
                continue
            if action == 'repair':
                new_pred = collapse(str(decision.get('predicate') or '')) or collapse(str(item.get('predicate') or ''))
                if settings.get('hierarchy_repair_predicates_only') and 'hierarchy_invalid' in {
                    collapse(str(c)).casefold() for c in (item.get('reason_codes') or [])
                }:
                    # Predicate/direction repair only — endpoints stay; gates re-check later.
                    new_subj = collapse(str(item.get('subject') or ''))
                    new_obj = collapse(str(item.get('object') or ''))
                else:
                    new_subj = collapse(str(decision.get('subject') or '')) or collapse(str(item.get('subject') or ''))
                    new_obj = collapse(str(decision.get('object') or '')) or collapse(str(item.get('object') or ''))
                source['subject'] = new_subj
                source['predicate'] = new_pred
                source['object'] = new_obj
                hint = collapse(str(decision.get('entity_type_hint') or ''))
                if hint and hint.casefold() in allowed_types:
                    source.setdefault('metadata', {})
                    if isinstance(source.get('metadata'), dict):
                        source['metadata']['llm_entity_type_hint'] = allowed_types[hint.casefold()]
                        source['metadata']['llm_hint_surface'] = new_obj
            meta = source.get('metadata') if isinstance(source.get('metadata'), dict) else {}
            meta = dict(meta)
            # Preserve prior evidence / section_context / offsets; only refresh quote.
            prior_evidence = meta.get('evidence') if isinstance(meta.get('evidence'), dict) else {}
            meta['evidence'] = {
                **prior_evidence,
                'source_text': quote,
                'source_file': source_file or prior_evidence.get('source_file'),
            }
            meta['evidence_recovery'] = 'llm_quarantine_resolver'
            meta.pop('graph_eligible', None)
            meta.pop('evidence_status', None)
            # Re-verify with the final repaired predicate/endpoints (hierarchy ≠ attribute).
            final_subj = collapse(str(source.get('subject') or ''))
            final_pred = collapse(str(source.get('predicate') or ''))
            final_obj = collapse(str(source.get('object') or ''))
            is_hierarchy = final_pred.casefold() in hierarchy_predicate_set(policy)
            is_attr = (
                not is_hierarchy
                and item.get('kind') == 'triplet'
                and (
                    _is_generic_attribute_predicate(final_pred, policy=policy)
                    or collapse(str(source.get('relation_category') or '')).casefold() == 'attribute'
                    or _looks_like_technical_attribute_claim(
                        final_pred,
                        final_obj,
                        evidence_quote=quote,
                        document_text=document_text,
                        policy=policy,
                    )
                )
            )
            # Attribute objects are properties — never seed entity mint from LLM hints.
            if is_attr:
                meta.pop('llm_entity_type_hint', None)
                meta.pop('llm_hint_surface', None)
            source['metadata'] = _attach_claim_evidence(
                meta,
                document_text,
                subject=final_subj,
                obj=final_obj,
                predicate=final_pred,
                attribute_mode=False if is_hierarchy else is_attr,
                source_file=source_file,
                policy=policy,
                owner_resolver=None,
            )
            corrected = collapse(str(source['metadata'].get('corrected_subject') or ''))
            if corrected:
                source['subject'] = corrected
            if is_attr:
                _apply_object_literal(
                    source,
                    obj=final_obj,
                    category='attribute',
                    typing_reason='property_not_entity',
                )
            # Count metadata mutations even when not yet graph-eligible so
            # quarantine can be re-synced with recovered (unverified) evidence.
            applied += 1
            if source['metadata'].get('graph_eligible') is not True:
                continue
            confirmed_applied += 1

    # Always rebuild quarantine after any recovery attempt (accurate reasons).
    sync = bool(
        _semantic_relation_recovery_settings(policy).get('sync_quarantine_after_recovery', True)
    )
    if not applied and not sync:
        return 0
    if applied:
        # Hinted primaries become entities only after independent YAML validation.
        attr_preds = {p.casefold() for p in attribute_predicate_hints(policy)}
        attr_object_keys: set[str] = set()
        for trip in bundle.triplets or []:
            if not isinstance(trip, dict):
                continue
            pred = collapse(str(trip.get('predicate') or ''))
            obj = collapse(str(trip.get('object') or ''))
            quote = _evidence_quote_for_row(trip)
            if (
                pred.casefold() in attr_preds
                or _is_generic_attribute_predicate(pred, policy=policy)
                or collapse(str(trip.get('relation_category') or '')).casefold() == 'attribute'
                or _looks_like_technical_attribute_claim(
                    pred, obj, evidence_quote=quote, document_text=document_text, policy=policy,
                )
            ):
                if obj:
                    attr_object_keys.add(obj.casefold())
        hinted: dict[str, str] = {}
        for pool in (bundle.relations, bundle.triplets):
            for row in pool:
                meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
                surface = collapse(str(meta.get('llm_hint_surface') or ''))
                label = collapse(str(meta.get('llm_entity_type_hint') or ''))
                if surface and label and surface.casefold() in (document_text or '').casefold():
                    hinted.setdefault(surface.casefold(), label)
        index = build_endpoint_resolution_index(bundle.entities, policy=policy)
        by_surface = index.get('by_surface') or {}
        for surface_key, label in hinted.items():
            if surface_key in by_surface:
                continue
            doc = document_text or ''
            idx = doc.casefold().find(surface_key)
            surface = doc[idx:idx + len(surface_key)] if idx >= 0 else surface_key
            validated = _validate_recovered_entity_proposal(
                surface,
                proposed_label=label,
                role='object',
                policy=policy,
                document_text=document_text,
                attribute_object=surface_key in attr_object_keys,
            )
            if validated is None:
                continue
            primary, subtype = validated
            eid = f'recovered_{hashlib.sha1(surface_key.encode("utf-8")).hexdigest()[:10]}'
            meta_out: dict[str, Any] = {
                'canonical_id': eid,
                'identifiers': [surface],
                'extraction_method': 'llm_quarantine_resolver',
                'source_file': source_file,
                'typing_reason': 'yaml_validated_recovery',
            }
            if subtype:
                meta_out['subtype'] = subtype
            bundle.entities.append({
                'id': eid,
                'text': surface,
                'label': primary,
                'confidence': 0.6,
                'metadata': meta_out,
            })
        recover_missing_endpoints(
            bundle, policy, document_text=document_text, source_file=source_file,
        )
        _scrub_invalid_recovered_entities(
            bundle, policy, document_text=document_text,
        )
        validate_extraction_graph(
            bundle, policy, source_file=source_file, document_text=document_text,
        )
    bundle.graph, bundle.quarantine = build_confirmed_graph(
        bundle, policy, source_file=source_file,
    )
    _diag_record(
        'quarantine_llm_resolver',
        candidates=len(selected),
        applied=applied,
        confirmed_applied=confirmed_applied,
    )
    return confirmed_applied


def _to_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
    return {'value': str(obj)}


def endpoint_text(value) -> str:
    if value is None:
        return ''
    if isinstance(value, dict):
        return collapse(str(value.get('text') or value.get('name') or ''))
    if hasattr(value, 'text'):
        return collapse(str(value.text))
    return collapse(str(value))


def _format_entity_lines(entities: list) -> str:
    lines = []
    for entity in entities or []:
        if isinstance(entity, dict):
            surface = collapse(str(entity.get('text') or ''))
            label = collapse(str(entity.get('label') or '')) or '(untyped)'
        else:
            surface = collapse(getattr(entity, 'text', '') or '')
            label = collapse(getattr(entity, 'label', '') or '') or '(untyped)'
        if surface:
            lines.append(f'- {surface} ({label})')
    return '\n'.join(lines) or '(none)'

# ---------------------------------------------------------------------------
# 12) FactExtractor — Semantica extract_* + local prompt / policy
# ---------------------------------------------------------------------------
class FactExtractor:
    """Thin wrapper: policy + prompt → Semantica NER / relations / triplets."""

    def __init__(
        self,
        *,
        prompt_dir: Path | None = None,
        policy: Policy | None = None,
        settings: Settings | None = None,
        min_confidence: float | None = None,
        max_retries: int | None = None,
    ):
        self.settings = settings or get_settings()
        self.policy = policy or load_policy()
        self.prompt_dir = Path(prompt_dir) if prompt_dir else PROMPT_DIR
        self.min_confidence = float(min_confidence if min_confidence is not None else self.settings.min_extraction_confidence)
        self.max_retries = int(max_retries if max_retries is not None else self.settings.llm_max_retries)
        self.provider = 'bedrock'
        self.llm_model = self.settings.bedrock_model_id
        self.type_hints = primary_type_hints(self.policy)

        install_guided_extract_hooks()
        # union: if LLM returns nothing, do not fall back to pattern PERSON/DATE junk
        self.ner = NamedEntityRecognizer(
            methods=['llm'],
            confidence_threshold=self.min_confidence,
            merge_overlapping=True,
            merge_strategy='union',
        )
        self.relation_extractor = RelationExtractor(method='llm', confidence_threshold=self.min_confidence)
        self.triplet_extractor = TripletExtractor(method='llm', include_provenance=True)
        self.event_detector = EventDetector(method='llm', ner_method='llm', provider=self.provider)
        self.coref_resolver = CoreferenceResolver(method='llm', provider=self.provider)
        self.duplicate_detector = DuplicateDetector(similarity_threshold=0.9, confidence_threshold=0.75)
        self.entity_merger = EntityMerger(preserve_provenance=True)

    def _dedupe_entities(
        self,
        entities: list,
        warnings: list[str],
    ) -> tuple[list, dict[str, str], dict[str, str]]:
        """Collapse near-duplicates but keep original representative Entity objects.

        Returns:
            kept entities, old_id→canonical_id map, surface→canonical surface map
        """
        rows: list[dict] = []
        identity_map: dict[str, str] = {}
        surface_map: dict[str, str] = {}
        for index, entity in enumerate(entities or []):
            text = collapse(getattr(entity, 'text', '') or '')
            if not text:
                continue
            label = collapse(getattr(entity, 'label', '') or '')
            eid = f'e{index}'
            rows.append({
                'id': eid,
                'name': text,
                'text': text,
                'type': label,
                'label': label,
                'confidence': float(getattr(entity, 'confidence', 0) or 0),
                'metadata': getattr(entity, 'metadata', None) or {},
                '_src': entity,
            })
            identity_map[eid] = eid
            surface_map[text.casefold()] = text
            meta = getattr(entity, 'metadata', None) or {}
            if isinstance(meta, dict):
                meta = dict(meta)
                meta['canonical_id'] = eid
                entity.metadata = meta
                for ident in _identifier_values(meta):
                    surface_map.setdefault(ident, text)

        if len(rows) < 2:
            dicts = [{
                'id': r['id'],
                'text': r['text'],
                'metadata': r.get('metadata') or {},
            } for r in rows]
            for key, val in _build_surface_map_from_entities(dicts, policy=self.policy).items():
                surface_map.setdefault(key, val)
            return list(entities or []), identity_map, surface_map

        try:
            groups = self.duplicate_detector.detect_duplicate_groups(rows)
            kept: list = []
            consumed: set[str] = set()
            for group in groups or []:
                members = list(getattr(group, 'entities', None) or [])
                if len(members) < 2:
                    continue
                # Never merge distinct systems/equipment on name similarity alone.
                if not _group_safe_to_merge(members):
                    continue

                op = self.entity_merger.merge_entity_group(members, strategy='keep_highest_confidence')
                merged = getattr(op, 'merged_entity', None) or {}
                mid = str(merged.get('id') or '')
                rep_row = next((m for m in members if str(m.get('id') or '') == mid), None)
                if rep_row is None:
                    rep_row = max(members, key=lambda m: float(m.get('confidence') or 0))
                representative = rep_row.get('_src')
                if representative is None:
                    continue

                canonical_id = str(rep_row.get('id') or mid)
                canonical_text = collapse(getattr(representative, 'text', '') or '')
                _apply_merge_to_representative(
                    representative,
                    members=members,
                    merged_entity=merged if isinstance(merged, dict) else {},
                    canonical_id=canonical_id,
                )

                for member in members:
                    eid = str(member.get('id') or '')
                    if eid:
                        consumed.add(eid)
                        identity_map[eid] = canonical_id
                    surface = collapse(str(member.get('text') or member.get('name') or ''))
                    if surface:
                        surface_map[surface.casefold()] = canonical_text
                    for alias in _alias_values(_meta_of(member)):
                        surface_map[alias] = canonical_text
                    for ident in _identifier_values(_meta_of(member)):
                        surface_map.setdefault(ident, canonical_text)

                kept.append(representative)

            for row in rows:
                if row['id'] not in consumed:
                    kept.append(row['_src'])
                    # Unmerged rows still expose identifiers for endpoint resolve.
                    canon = collapse(str(row.get('text') or ''))
                    if canon:
                        for ident in _identifier_values(_meta_of(row)):
                            surface_map.setdefault(ident, canon)
            # Enrich with unique engineering-tag keys after keep-set is known.
            kept_dicts = []
            for ent in kept:
                kept_dicts.append({
                    'id': collapse(str((getattr(ent, 'metadata', None) or {}).get('canonical_id') or '')),
                    'text': collapse(getattr(ent, 'text', '') or ''),
                    'metadata': getattr(ent, 'metadata', None) or {},
                })
            # Rebuild ids for index from metadata.canonical_id assigned earlier.
            for idx, ent in enumerate(kept):
                meta = getattr(ent, 'metadata', None) or {}
                if not isinstance(meta, dict):
                    meta = {}
                if not meta.get('canonical_id'):
                    meta = dict(meta)
                    meta['canonical_id'] = f'e{idx}'
                    ent.metadata = meta
                kept_dicts[idx]['id'] = collapse(str(meta.get('canonical_id') or f'e{idx}'))
            extra = _build_surface_map_from_entities(kept_dicts, policy=self.policy)
            for key, val in extra.items():
                surface_map.setdefault(key, val)
            return kept, identity_map, surface_map
        except Exception as exc:
            warnings.append(f'deduplicate/merge skipped: {exc}')
            return list(entities or []), identity_map, surface_map

    def _unified_prompt(self, *, task: str, text: str, entities: list | None = None, source_file: str | None = None) -> str:
        hints = self.type_hints
        relation_hints = relation_predicate_hints(self.policy)
        attribute_hints = attribute_predicate_hints(self.policy)
        triplet_hints = [h for h in relation_hints if h.casefold() not in {a.casefold() for a in attribute_hints}]
        task_key = collapse(task or 'entities').casefold()
        evidence_schema = (
            '{"source_text": "", "source_file": "' + (source_file or '') + '", '
            '"start_char": null, "end_char": null}'
        )

        # NER uses a focused EntitiesResponse prompt — the full unified template
        # over-attends to page banners and under-extracts tagged assets on Nova Pro.
        if task_key in {'entity', 'entities', 'ner'}:
            type_line = ', '.join(hints) or '(none)'
            non_line = ', '.join(non_entity_type_hints(self.policy)) or '(none)'
            # Keep schema example free of real path tokens — embedding source_file
            # (e.g. Annexure / document numbers) causes Nova to "extract" those
            # strings from the schema instead of tagged assets in SOURCE.
            ner_evidence_schema = (
                '{"source_text": "", "source_file": "", '
                '"start_char": null, "end_char": null}'
            )
            ontology = ontology_hints(self.policy, compact=True)
            return (
                'TASK = ENTITIES only (NER).\n'
                'Return Semantica EntitiesResponse JSON only:\n'
                '{"entities": [{"text": "", "label": "", "confidence": 0.0, '
                '"metadata": {"kind": "entity", "subtype": "", "identifiers": '
                '[{"value": "", "role": "identity"}], "aliases": [], "properties": {}, '
                f'"evidence": {ner_evidence_schema}'
                '}}]}\n\n'
                'Rules:\n'
                '- Extract EVERY independently identifiable entity supported by the SOURCE '
                'text below — not from this schema example, filenames, or instructions.\n'
                f'- Use TYPE_HINTS primaries: {type_line}\n'
                f'- Do not use NON_ENTITY labels: {non_line}\n'
                '- Classify each entity from source context and the YAML definitions / '
                'disambiguation / non-entity policy below; never invent a type from name '
                'shape alone or from memorised example phrases.\n'
                '- Emit a node only for independently identifiable referents; treat '
                'scalar properties, alias-only identifiers, relation predicates, noise, '
                'field labels, and incomplete headings per the Non-entity policy in YAML guidance.\n'
                '- Primary label: predict from TYPE_HINTS + YAML entity definitions (source role).\n'
                '- Subtype: predict from YAML Recommended subtypes / disambiguation when source '
                'supports it; never use bare subtype Document; never invent subtypes.\n'
                '- Omit field-label captions per Non-entity policy (keep identifiable values only).\n'
                '- Prefer one canonical entity per named section title.\n'
                '- Verbatim source spans only; never invent names or identifiers.\n'
                '- Extract tagged assets in table data rows and body text; column headers '
                'alone are not entities.\n'
                '- Ignore repeated page headers/footers/metadata banners when they recur by '
                'structure; never skip technical, commercial, specification, or tender-data '
                'tables — if unsure whether a block is boilerplate, treat it as content.\n'
                '- Prefer completeness for clearly named technical assets; omit only when '
                'evidence is insufficient.\n'
                '- One entity per distinct real-world object; keep identifiers exact.\n'
                '- PK middle-segment tags → System + EngineeringPackage (never Equipment); '
                'one canonical entity per PK tag.\n'
                '- REQUIRED: set metadata.evidence.source_text to the shortest verbatim '
                'supporting span from SOURCE for each entity; leave source_file empty '
                'and start_char/end_char null (filled downstream).\n'
                '- Never emit placeholder tokens, schema field names, or generic demo names '
                '(e.g. sample persons/orgs) that are not verbatim in SOURCE.\n\n'
                'YAML typing guidance:\n'
                f'{ontology}\n\n'
                'SOURCE\n'
                '<<CHUNK_TEXT>>'
            ).strip()

        if task_key in {'relation', 'relations'}:
            task_focus = (
                'TASK = RELATIONS only (entity–entity).\n'
                'Prefer Known entities below as endpoints when present; do not invent '
                'endpoints, but extract every source-supported entity–entity relation '
                'among known or clearly named entities.\n'
                'Preserve correct predicate direction from the source.\n'
                'Emit a relation only when one contiguous SOURCE span supports '
                'subject–predicate–object (including direction); omit if missing.\n'
                'REQUIRED: metadata.evidence.source_text = shortest verbatim span '
                'supporting subject, predicate meaning, and object (not endpoint '
                'co-mention alone); leave offsets null.\n'
                'Do not infer package/vendor → annexure-index / TOC containment or '
                'supply (includes/hasEquipment/partOf/provides) without an explicit span.\n'
                'Do not assign general QA / inspection clauses to equipment outside '
                'that clause\'s explicit ownership span.\n'
                'Do not use requirement/contract prose sentences as relation objects.\n'
                'Return Semantica RelationsResponse JSON:\n'
                '{"relations": [{"subject": "", "predicate": "", "object": "", '
                '"confidence": 0.0, "metadata": {"evidence": ' + evidence_schema + '}}]}'
            )
        elif task_key in {'triplet', 'triplets'}:
            task_focus = (
                'TASK = TRIPLETS (entity–entity and entity–attribute).\n'
                'Use Known entities below as preferred subjects/objects when present.\n'
                'Extract all technical attributes from tables.\n'
                'Create one atomic triplet per attribute-value pair.\n'
                'Use the source field label as the predicate when present; '
                'put only the exact value and unit in the object.\n'
                'Do not use generic has property for labeled table fields.\n'
                'REQUIRED: metadata.evidence.source_text = shortest verbatim value '
                'span (and field label when present). Do not concatenate a distant '
                'section heading into the quote; ownership may use section context.\n'
                'For entity–entity triplets, quote must support predicate meaning '
                'and direction, not co-mention alone.\n'
                'Preserve correct system, value, and unit.\n'
                'Do not skip attributes when relations exist.\n'
                'Return Semantica TripletsResponse JSON:\n'
                '{"triplets": [{"subject": "", "predicate": "", "object": "", '
                '"confidence": 0.0, "metadata": {"evidence": ' + evidence_schema + '}}]}'
            )
        else:
            task_focus = f'TASK = {task_key}. Return valid JSON only.'

        try:
            return load_prompt(
                self.prompt_dir,
                'unified_extraction',
                TASK_FOCUS=task_focus,
                TYPE_HINTS=', '.join(hints) or '(none)',
                NON_ENTITY_TYPE_HINTS=', '.join(non_entity_type_hints(self.policy)) or '(none)',
                ONTOLOGY_HINTS=ontology_hints(self.policy, compact=False),
                ENTITIES=_format_entity_lines(entities or []),
                RELATION_HINTS=', '.join(relation_hints) or '(open vocabulary)',
                TRIPLET_HINTS=', '.join(triplet_hints) or '(open vocabulary)',
                ATTRIBUTE_PREDICATE_HINTS=', '.join(attribute_hints) or '(open vocabulary)',
                SOURCE_FILE=source_file or '',
                # Placeholder only — filled per Semantica chunk in _augment().
                # Embedding the full document here breaks large-file chunking.
                TEXT='<<CHUNK_TEXT>>',
            ).strip()
        except FileNotFoundError:
            return ''

    def _llm_opts(self, *, kind: str, text: str, entities: list | None = None, source_file: str | None = None) -> dict:
        opts: dict[str, Any] = {
            'method': 'llm',
            'provider': self.provider,
            'silent_fail': False,
            'max_retries': self.max_retries,
            'max_tokens': int(self.settings.extraction_max_tokens or self.settings.bedrock_max_tokens or 8192),
            'max_text_length': int(self.settings.extraction_chunk_chars or 12000),
        }
        if self.llm_model:
            opts['llm_model'] = self.llm_model
        if self.type_hints:
            opts['entity_types'] = self.type_hints
        if source_file:
            opts['source_file'] = source_file

        if kind == 'entity':
            guidance = self._unified_prompt(task='entities', text=text, source_file=source_file)
            if guidance:
                opts['domain_instructions'] = guidance
            # Serialize Semantica chunk workers so BedrockProvider hooks / typed
            # store stay coherent (ThreadPoolExecutor does not copy guidance state).
            opts['max_workers'] = 1
        elif kind == 'relation':
            relation_hints = relation_predicate_hints(self.policy)
            if relation_hints:
                opts['relation_types'] = relation_hints[:40]
            guidance = self._unified_prompt(task='relations', text=text, entities=entities, source_file=source_file)
            if guidance:
                opts['domain_instructions'] = guidance
            opts['max_workers'] = 1
        elif kind == 'triplet':
            triplet_hints = relation_predicate_hints(self.policy) + attribute_predicate_hints(self.policy)
            if triplet_hints:
                opts['triplet_types'] = list(dict.fromkeys(triplet_hints))[:40]
            guidance = self._unified_prompt(task='triplets', text=text, entities=entities, source_file=source_file)
            if guidance:
                opts['domain_instructions'] = guidance
            opts['max_workers'] = 1
        return opts

    def extract(self, text: str, *, source_file: str | None = None) -> ExtractionBundle:
        """Run Semantica entity → relation → triplet extraction on one text."""
        bundle = ExtractionBundle()
        if not collapse(text):
            return bundle

        _diag_begin()
        max_chars = int(self.settings.extraction_chunk_chars or 12000)
        chunk_spans = section_aware_chunk_spans(text, max_chars)
        _DOC_NER_CONTEXT.text = text
        _DOC_NER_CONTEXT.spans = detect_repeated_boilerplate_spans(text, policy=self.policy)
        _DOC_NER_CONTEXT.parent_heading = ''
        _DOC_NER_CONTEXT.chunk_start = 0
        _diag_record(
            'extract_start',
            source_file=source_file or '',
            doc_chars=len(text or ''),
            will_chunk_entities=len(chunk_spans) > 1,
            section_chunks=len(chunk_spans),
            boilerplate_omit=sum(
                1 for s in (_DOC_NER_CONTEXT.spans or [])
                if s.get('action') == 'omit_repeat'
            ),
        )

        entity_opts = self._llm_opts(kind='entity', text=text, source_file=source_file)
        entities: list = []
        relations: list = []
        triplets: list = []
        try:
            try:
                for span in chunk_spans:
                    _DOC_NER_CONTEXT.chunk_start = int(span['start'])
                    _DOC_NER_CONTEXT.parent_heading = span.get('parent_heading') or ''
                    opts = dict(entity_opts)
                    # Prevent Semantica from re-splitting our section-aware window.
                    opts['max_text_length'] = max(len(span['text']) + 64, max_chars)
                    chunk_entities = list(self.ner.extract_entities(span['text'], **opts) or [])
                    entities.extend(
                        _postprocess_chunk_extract(
                            chunk_entities,
                            full_text=text,
                            source_file=source_file,
                            span=span,
                            policy=self.policy,
                            kind='entity',
                        )
                    )
            except Exception as exc:
                bundle.warnings.append(f'extract_entities failed: {exc}')
                _diag_record('after_entities_llm', entities=0, error=str(exc))

            print(f'[stage] after_metadata_merge entities={len(entities)}')
            _diag_record(
                'after_metadata_merge',
                entities=len(entities),
                section_chunks=len(chunk_spans),
                chunk_methods=sorted({
                    str(s.get('method') or '') for s in chunk_spans if s.get('method')
                }),
            )

            # Snapshot LLM NER before coref / dedupe / kind filtering.
            for entity in entities:
                row = _to_dict(entity)
                meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else (
                    getattr(entity, 'metadata', None) or {}
                )
                if isinstance(meta, dict):
                    meta = _attach_verified_evidence(meta, text, source_file=source_file)
                bundle.raw_entities.append({
                    'text': collapse(str(row.get('text') or getattr(entity, 'text', '') or '')),
                    'label': collapse(str(row.get('label') or getattr(entity, 'label', '') or '')),
                    'confidence': float(
                        row.get('confidence')
                        if row.get('confidence') is not None
                        else getattr(entity, 'confidence', 0) or 0
                    ),
                    'start_char': row.get('start_char', getattr(entity, 'start_char', None)),
                    'end_char': row.get('end_char', getattr(entity, 'end_char', None)),
                    'metadata': meta if isinstance(meta, dict) else {},
                })

            try:
                before_coref = len(entities)
                chains = list(
                    self.coref_resolver.resolve_coreferences(
                        text, entities=entities, **entity_opts,
                    ) or []
                )
                dropped_coref: list[str] = []
                if chains:
                    # Soft coref: drop only explicit aliases from LLM metadata.
                    alias_of: dict[str, str] = {}
                    for ent in entities:
                        canon = collapse(getattr(ent, 'text', '') or '').casefold()
                        meta = getattr(ent, 'metadata', None) or {}
                        for alias in meta.get('aliases') or []:
                            alias_key = collapse(str(alias or '')).casefold()
                            if alias_key and canon:
                                alias_of[alias_key] = canon
                    drop: set[str] = set()
                    for chain in chains:
                        rep = getattr(chain, 'representative', None)
                        rep_text = collapse(getattr(rep, 'text', '') or '').casefold()
                        for mention in getattr(chain, 'mentions', None) or []:
                            surface = collapse(getattr(mention, 'text', '') or '').casefold()
                            if surface and surface != rep_text and alias_of.get(surface) == rep_text:
                                drop.add(surface)
                                dropped_coref.append(f'alias_of_canonical:{surface}->{rep_text}')
                    entities = [
                        e for e in entities
                        if collapse(getattr(e, 'text', '') or '').casefold() not in drop
                    ]
                print(f'[stage] after_coreference entities={len(entities)}')
                _diag_record(
                    'after_coreference',
                    input=before_coref,
                    kept=len(entities),
                    dropped_count=len(dropped_coref),
                    dropped=dropped_coref,
                )
            except Exception as exc:
                bundle.warnings.append(f'resolve_coreferences skipped: {exc}')
                print(f'[stage] after_coreference entities={len(entities)} (skipped)')
                _diag_record('after_coreference', kept=len(entities), skipped=str(exc))

            before_dedupe = len(entities)
            entities, identity_map, surface_map = self._dedupe_entities(entities, bundle.warnings)
            bundle.identity_map = identity_map
            print(f'[stage] after_deduplication entities={len(entities)}')
            _diag_record(
                'after_deduplication',
                input=before_dedupe,
                kept=len(entities),
                identity_map_size=len(identity_map),
            )

            relation_opts = self._llm_opts(
                kind='relation', text=text, entities=entities, source_file=source_file,
            )
            try:
                for span in chunk_spans:
                    _DOC_NER_CONTEXT.chunk_start = int(span['start'])
                    _DOC_NER_CONTEXT.parent_heading = span.get('parent_heading') or ''
                    opts = dict(relation_opts)
                    opts['max_text_length'] = max(len(span['text']) + 64, max_chars)
                    chunk_relations = list(
                        self.relation_extractor.extract_relations(
                            span['text'], entities=entities, **opts,
                        ) or []
                    )
                    relations.extend(
                        _postprocess_chunk_extract(
                            chunk_relations,
                            full_text=text,
                            source_file=source_file,
                            span=span,
                            policy=self.policy,
                            kind='relation',
                        )
                    )
                _diag_record(
                    'after_relations_llm',
                    relations=len(relations),
                    known_entities=len(entities),
                    section_chunks=len(chunk_spans),
                )
            except Exception as exc:
                bundle.warnings.append(f'extract_relations failed: {exc}')
                _diag_record('after_relations_llm', relations=0, error=str(exc))

            triplet_opts = self._llm_opts(
                kind='triplet', text=text, entities=entities, source_file=source_file,
            )
            try:
                for span in chunk_spans:
                    _DOC_NER_CONTEXT.chunk_start = int(span['start'])
                    _DOC_NER_CONTEXT.parent_heading = span.get('parent_heading') or ''
                    opts = dict(triplet_opts)
                    opts['max_text_length'] = max(len(span['text']) + 64, max_chars)
                    chunk_triplets = list(
                        self.triplet_extractor.extract_triplets(
                            span['text'], entities=entities, relations=relations, **opts,
                        ) or []
                    )
                    triplets.extend(
                        _postprocess_chunk_extract(
                            chunk_triplets,
                            full_text=text,
                            source_file=source_file,
                            span=span,
                            policy=self.policy,
                            kind='triplet',
                        )
                    )
                _diag_record(
                    'after_triplets_llm',
                    triplets=len(triplets),
                    section_chunks=len(chunk_spans),
                )
            except Exception as exc:
                bundle.warnings.append(f'extract_triplets failed: {exc}')
                _diag_record('after_triplets_llm', triplets=0, error=str(exc))

            events = []
            try:
                events = list(self.event_detector.detect_events(text, **entity_opts) or [])
            except Exception as exc:
                bundle.warnings.append(f'detect_events skipped: {exc}')

            kind_drops: list[str] = []
            for entity in entities:
                row = _to_dict(entity)
                surface = collapse(str(row.get('text') or getattr(entity, 'text', '') or ''))
                if not surface:
                    kind_drops.append('empty_surface')
                    continue
                meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
                meta = _attach_verified_evidence(meta, text, source_file=source_file)
                refs = []
                for ref in meta.get('source_references') or []:
                    if not isinstance(ref, dict):
                        continue
                    verified = _verify_evidence(ref, text, source_file=source_file)
                    if not verified:
                        continue
                    verified['from_entity_id'] = ref.get('from_entity_id')
                    refs.append(verified)
                if refs:
                    meta['source_references'] = refs
                kind = collapse(str(meta.get('kind') or '')).casefold()
                if kind and kind != 'entity':
                    kind_drops.append(f'non_entity_kind:{surface}:{kind}')
                    continue
                if _is_field_label_non_entity(surface, text, policy=self.policy):
                    kind_drops.append(f'field_label:{surface}')
                    continue
                entity_id = collapse(str(meta.get('canonical_id') or '')) or None
                bundle.entities.append({
                    'id': entity_id,
                    'text': surface,
                    'label': collapse(str(row.get('label') or getattr(entity, 'label', '') or '')),
                    'confidence': float(
                        row.get('confidence')
                        if row.get('confidence') is not None
                        else getattr(entity, 'confidence', 0) or 0
                    ),
                    'metadata': meta,
                })
            _diag_record(
                'after_entity_serialize',
                input=len(entities),
                kept=len(bundle.entities),
                dropped_count=len(kind_drops),
                dropped=kind_drops,
            )

            # YAML-driven identity merge before relation/triplet endpoint binding.
            identity_merge_map = merge_identity_duplicate_entities(
                bundle,
                self.policy,
                document_text=text,
                source_file=source_file,
            )

            endpoint_index = build_endpoint_resolution_index(bundle.entities, policy=self.policy)
            surface_map = _build_surface_map_from_entities(bundle.entities, policy=self.policy)

            attr_preds = {p.casefold() for p in attribute_predicate_hints(self.policy)}
            rel_status = {'verified': 0, 'unverified': 0, 'missing': 0}
            for rel in relations:
                row = _to_dict(rel)
                subject = _canonicalize_endpoint(
                    endpoint_text(
                        row.get('subject')
                        if row.get('subject') is not None
                        else getattr(rel, 'subject', None)
                    ),
                    surface_map,
                    index=endpoint_index,
                    policy=self.policy,
                )
                predicate = collapse(str(row.get('predicate') or getattr(rel, 'predicate', '') or ''))
                obj = _canonicalize_endpoint(
                    endpoint_text(
                        row.get('object')
                        if row.get('object') is not None
                        else getattr(rel, 'object', None)
                    ),
                    surface_map,
                    index=endpoint_index,
                    policy=self.policy,
                )
                if subject and predicate and obj:
                    meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
                    meta = _attach_claim_evidence(
                        meta,
                        text,
                        subject=subject,
                        obj=obj,
                        predicate=predicate,
                        attribute_mode=False,
                        source_file=source_file,
                        policy=self.policy,
                    )
                    status = collapse(str(meta.get('evidence_status') or 'missing')).casefold()
                    if status in rel_status:
                        rel_status[status] += 1
                    if not meta.get('graph_eligible'):
                        bundle.warnings.append(
                            f'unverified/ineligible relation kept for review: '
                            f'{subject!r} -[{predicate}]-> {obj!r} '
                            f'(status={meta.get("evidence_status")})'
                        )
                    bundle.relations.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'confidence': float(
                            row.get('confidence')
                            if row.get('confidence') is not None
                            else getattr(rel, 'confidence', 0) or 0
                        ),
                        'metadata': meta,
                    })
            _diag_record(
                'after_relation_serialize',
                input=len(relations),
                kept=len(bundle.relations),
                **{f'evidence_{k}': v for k, v in rel_status.items()},
            )

            trip_status = {'verified': 0, 'unverified': 0, 'missing': 0}
            for trip in triplets:
                row = _to_dict(trip)
                subject = _canonicalize_endpoint(
                    collapse(str(row.get('subject') or getattr(trip, 'subject', '') or '')),
                    surface_map,
                    index=endpoint_index,
                    policy=self.policy,
                )
                predicate = collapse(str(row.get('predicate') or getattr(trip, 'predicate', '') or ''))
                raw_obj = collapse(str(row.get('object') or getattr(trip, 'object', '') or ''))
                evidence_quote = ''
                row_meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
                if isinstance(row_meta.get('evidence'), dict):
                    evidence_quote = str(row_meta['evidence'].get('source_text') or '')
                is_attr = (
                    predicate.casefold() in attr_preds
                    or _is_generic_attribute_predicate(predicate, policy=self.policy)
                    or _looks_like_technical_attribute_claim(
                        predicate,
                        raw_obj,
                        evidence_quote=evidence_quote,
                        document_text=text,
                        policy=self.policy,
                    )
                )
                if is_attr:
                    obj = raw_obj
                else:
                    obj = _canonicalize_endpoint(
                        raw_obj, surface_map, index=endpoint_index, policy=self.policy,
                    )
                if subject and predicate and obj:
                    meta = row_meta if isinstance(row_meta, dict) else {}
                    section_settings = _section_aware_evidence_settings(self.policy)
                    owner_resolver = (
                        _default_llm_owner_resolver
                        if section_settings.get('llm_owner_fallback', True)
                        else None
                    )
                    meta = _attach_claim_evidence(
                        meta,
                        text,
                        subject=subject,
                        obj=obj,
                        predicate=predicate,
                        attribute_mode=is_attr,
                        source_file=source_file,
                        policy=self.policy,
                        owner_resolver=owner_resolver,
                    )
                    corrected = collapse(str(meta.get('corrected_subject') or ''))
                    if corrected:
                        subject = corrected
                    status = collapse(str(meta.get('evidence_status') or 'missing')).casefold()
                    if status in trip_status:
                        trip_status[status] += 1
                    if not meta.get('graph_eligible'):
                        bundle.warnings.append(
                            f'unverified/ineligible triplet kept for review: '
                            f'{subject!r} -[{predicate}]-> {obj!r} '
                            f'(status={meta.get("evidence_status")}, '
                            f'predicate_quality={meta.get("predicate_quality")})'
                        )
                    bundle.triplets.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'confidence': float(
                            row.get('confidence')
                            if row.get('confidence') is not None
                            else getattr(trip, 'confidence', 0) or 0
                        ),
                        'metadata': meta,
                    })
            _diag_record(
                'after_triplet_serialize',
                input=len(triplets),
                kept=len(bundle.triplets),
                **{f'evidence_{k}': v for k, v in trip_status.items()},
            )

            for event in events:
                bundle.events.append(_to_dict(event))

            recover_missing_endpoints(
                bundle,
                self.policy,
                document_text=text,
                source_file=source_file,
            )
            bundle.warnings.extend(
                validate_extraction_graph(
                    bundle,
                    self.policy,
                    source_file=source_file,
                    document_text=text,
                )
            )
            # Remap any pre-merge endpoint ids; surfaces stay for audit/self-loop detection.
            _redirect_bundle_endpoints_after_identity_merge(
                bundle, identity_merge_map or bundle.identity_map,
            )
            bundle.graph, bundle.quarantine = build_confirmed_graph(
                bundle, self.policy, source_file=source_file,
            )
            # Source-grounded semantic recovery (YAML contracts + evidence index).
            # Independent of LLM; rebuilds quarantine when sync is enabled.
            recover_semantic_relations(
                bundle,
                self.policy,
                document_text=text,
                source_file=source_file,
            )
            resolve_quarantine_with_llm(
                bundle,
                self.policy,
                document_text=text,
                source_file=source_file,
            )
            _diag_record(
                'extract_complete',
                raw_entities=len(bundle.raw_entities),
                entities=len(bundle.entities),
                relations=len(bundle.relations),
                triplets=len(bundle.triplets),
                graph_nodes=(bundle.graph or {}).get('node_count', 0),
                graph_edges=(bundle.graph or {}).get('edge_count', 0),
                quarantine_relations=(bundle.quarantine or {}).get('relation_count', 0),
                quarantine_triplets=(bundle.quarantine or {}).get('triplet_count', 0),
                warnings=len(bundle.warnings),
            )
            bundle.diagnostics = _diag_snapshot()
            return bundle
        finally:
            _DOC_NER_CONTEXT.text = None
            _DOC_NER_CONTEXT.spans = None
            _DOC_NER_CONTEXT.parent_heading = ''
            _DOC_NER_CONTEXT.chunk_start = 0

# ---------------------------------------------------------------------------
# 13) CLI — read inputs/, run FactExtractor, write outputs/
# ---------------------------------------------------------------------------
def canonicalize_source(source: str, *, root: Path | None = None) -> str:
    """Stable document identity across absolute/relative Windows/Unix path aliases.

    Prefer `inputs/<file>` when the path contains an `inputs` segment; otherwise
    prefer a path relative to the entity-relationship root (or its parent).
    Original paths should be retained separately as provenance aliases.
    """
    raw = collapse(str(source or '')).replace('\\', '/')
    if not raw:
        return 'unknown'
    base = (root or ROOT).resolve()
    parts = [p for p in raw.split('/') if p not in ('', '.')]
    if 'inputs' in [p.casefold() for p in parts]:
        idx = next(i for i, p in enumerate(parts) if p.casefold() == 'inputs')
        return '/'.join(parts[idx:])
    try:
        path = Path(raw)
        candidate = path.resolve() if path.is_absolute() else (base / path).resolve()
    except Exception:
        return '/'.join(parts)
    for anchor in (base, base.parent):
        try:
            return candidate.relative_to(anchor.resolve()).as_posix()
        except Exception:
            continue
    return candidate.as_posix()


def discover_inputs(path: Path | None = None) -> list[Path]:
    if path is not None:
        target = path if path.is_absolute() else ROOT / path
        if target.is_file():
            return [target]
        if target.is_dir():
            return sorted(target.glob('*.md'))
        raise FileNotFoundError(f'Input not found: {target}')
    if not INPUT_DIR.is_dir():
        return []
    return sorted(INPUT_DIR.glob('*.md'))


def extract_file(path: Path, extractor: FactExtractor) -> dict:
    text = path.read_text(encoding='utf-8')
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        source_original = resolved.relative_to(ROOT).as_posix()
    elif resolved.is_relative_to(ROOT.parent):
        source_original = resolved.relative_to(ROOT.parent).as_posix()
    else:
        source_original = str(resolved).replace('\\', '/')
    source = canonicalize_source(source_original)
    print(f'[extract] {path.name} ({len(text)} chars)', flush=True)
    bundle = extractor.extract(text, source_file=source)
    return {
        'source': source,
        'source_original': source_original,
        'raw_entity_count': len(bundle.raw_entities),
        'entity_count': len(bundle.entities),
        'relation_count': len(bundle.relations),
        'triplet_count': len(bundle.triplets),
        'raw_entities': bundle.raw_entities,
        'entities': bundle.entities,
        'relations': bundle.relations,
        'triplets': bundle.triplets,
        'events': bundle.events,
        'identity_map': bundle.identity_map,
        'graph': bundle.graph,
        'quarantine': bundle.quarantine,
        'warnings': bundle.warnings,
        'diagnostics': bundle.diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Semantica FactExtractor over entity-relationship/inputs')
    parser.add_argument('--file', type=Path, default=None, help='Single .md file or folder (default: inputs/)')
    parser.add_argument('--out', type=Path, default=OUTPUT_DIR, help='Output directory for JSON')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    settings = get_settings()
    has_env_creds = bool(
        settings.bedrock_api_key
        or os.environ.get('AWS_BEARER_TOKEN_BEDROCK')
        or os.environ.get('AWS_ACCESS_KEY_ID')
    )
    has_shared_creds = (Path.home() / '.aws' / 'credentials').is_file()
    if not has_env_creds and not has_shared_creds:
        print('Missing Bedrock credentials. Set BEDROCK_API_KEY in entity-relationship/.env')
        return 1

    policy = load_policy()
    extractor = FactExtractor(policy=policy, settings=settings)
    files = discover_inputs(args.file)
    if not files:
        print(f'No .md files found under {INPUT_DIR}')
        return 1

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        result = extract_file(path, extractor)
        dest = out_dir / f'{path.stem}_extraction.json'
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        graph = result.get('graph') or {}
        quarantine = result.get('quarantine') or {}
        print(
            f'[done] {path.name} → {dest.name} '
            f'(raw_entities={result["raw_entity_count"]}, entities={result["entity_count"]}, '
            f'relations={result["relation_count"]}, triplets={result["triplet_count"]}, '
            f'graph_edges={graph.get("edge_count", 0)}, '
            f'quarantine={quarantine.get("relation_count", 0) + quarantine.get("triplet_count", 0)}, '
            f'warnings={len(result["warnings"])})',
                flush=True,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
