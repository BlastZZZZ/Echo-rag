import re
from typing import Dict, List, Sequence, Tuple

from .utils.misc_utils import text_processing


def _normalize_slot_text(text: str) -> str:
    processed = text_processing(text)
    if not isinstance(processed, str):
        return ""
    normalized = processed.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def _collect_phrase_hits(
    normalized_text: str,
    phrase_map: Sequence[Tuple[str, str]],
) -> List[str]:
    padded_text = f" {normalized_text} "
    hits: List[str] = []
    seen = set()
    for phrase, slot_value in phrase_map:
        normalized_phrase = _normalize_slot_text(phrase)
        if not normalized_phrase:
            continue
        if f" {normalized_phrase} " not in padded_text:
            continue
        if slot_value in seen:
            continue
        seen.add(slot_value)
        hits.append(slot_value)
    return hits


def _append_unique(values: List[str], candidate: str) -> None:
    if candidate and candidate not in values:
        values.append(candidate)


COMPARISON_OR_STATE_PHRASES: Sequence[Tuple[str, str]] = (
    ("born earlier", "earlier"),
    ("born later", "later"),
    ("died earlier", "earlier"),
    ("died later", "later"),
    ("born first", "earlier"),
    ("died first", "earlier"),
    ("born second", "later"),
    ("died second", "later"),
    ("older than", "older"),
    ("younger than", "younger"),
    ("older", "older"),
    ("younger", "younger"),
    ("earlier", "earlier"),
    ("later", "later"),
    ("lowest batting average", "lowest"),
    ("lowest", "lowest"),
    ("highest", "highest"),
    ("largest", "largest"),
    ("smallest", "smallest"),
    ("wettest", "wettest"),
    ("most populous", "most-populous"),
    ("most games", "most-games"),
    ("majority party", "majority"),
    ("minority party", "minority"),
    ("same nationality", "same-nationality"),
    ("same country", "same-country"),
    ("same state", "same-state"),
    ("second pick", "second-pick"),
    ("first pick", "first-pick"),
    ("top ranking", "top-ranking"),
    ("only group larger", "only-larger"),
    ("gain control", "control-gained"),
    ("gained control", "control-gained"),
)

ROLE_IDENTITY_PHRASES: Sequence[Tuple[str, str]] = (
    ("director", "director"),
    ("direct by", "director"),
    ("composer", "composer"),
    ("music by", "composer"),
    ("producer", "producer"),
    ("produced by", "producer"),
    ("writer", "writer"),
    ("written by", "writer"),
    ("designer", "designer"),
    ("designed by", "designer"),
    ("performer", "performer"),
    ("performed by", "performer"),
    ("governor", "governor"),
    ("speaker", "speaker"),
    ("archbishop", "archbishop"),
    ("explorer", "explorer"),
    ("husband", "husband"),
    ("wife", "wife"),
    ("father", "father"),
    ("mother", "mother"),
    ("child in law", "child-in-law"),
    ("child of", "child"),
    ("son of", "child"),
    ("daughter of", "child"),
    ("queen to", "spouse"),
    ("king to", "spouse"),
    ("married to", "spouse"),
    ("step of", "step-relation"),
)

TIME_LIFECYCLE_PHRASES: Sequence[Tuple[str, str]] = (
    ("born", "born"),
    ("died", "died"),
    ("death", "died"),
    ("created", "created"),
    ("established", "established"),
    ("abolished", "abolished"),
    ("before death", "before-death"),
    ("before his death", "before-death"),
    ("before her death", "before-death"),
    ("after death", "after-death"),
    ("reach", "reached"),
    ("reached", "reached"),
)

GEO_SPATIAL_PHRASES: Sequence[Tuple[str, str]] = (
    ("north of", "north-of"),
    ("south of", "south-of"),
    ("east of", "east-of"),
    ("west of", "west-of"),
    ("adjacent to", "adjacent"),
    ("adjacent", "adjacent"),
    ("shares a border", "border"),
    ("border", "border"),
    ("between", "between"),
    ("near", "near"),
    ("empty into", "empties-into"),
)

ROLE_IDENTITY_NORMALIZATION = {
    "husband": "spouse",
    "wife": "spouse",
    "queen": "spouse",
    "king": "spouse",
    "queen/king spouse": "spouse",
    "spouse": "spouse",
}

COMPARISON_AXIS_PHRASES: Sequence[Tuple[str, str]] = (
    ("nationality", "nationality"),
    ("citizen of", "nationality"),
    ("citizenship", "nationality"),
    ("same country", "nationality"),
    ("same nationality", "nationality"),
    ("majority", "control"),
    ("minority", "control"),
    ("gain control", "control"),
    ("gained control", "control"),
    ("control", "control"),
    ("population", "population"),
    ("most populous", "population"),
    ("largest populated", "population"),
    ("largest", "population"),
    ("wettest", "precipitation"),
    ("batting average", "batting-average"),
    ("draft", "draft-order"),
    ("pick", "draft-order"),
    ("games", "games-played"),
    ("group larger", "organization-size"),
    ("record label", "organization-size"),
)


def extract_support_slot_inventory(text: str) -> Dict[str, List[str]]:
    normalized_text = _normalize_slot_text(text)
    if not normalized_text:
        return {
            "comparison_or_state_slots": [],
            "role_identity_slots": [],
            "time_lifecycle_slots": [],
            "geo_spatial_slots": [],
        }

    return {
        "comparison_or_state_slots": _collect_phrase_hits(
            normalized_text,
            COMPARISON_OR_STATE_PHRASES,
        ),
        "role_identity_slots": _collect_phrase_hits(
            normalized_text,
            ROLE_IDENTITY_PHRASES,
        ),
        "time_lifecycle_slots": _collect_phrase_hits(
            normalized_text,
            TIME_LIFECYCLE_PHRASES,
        ),
        "geo_spatial_slots": _collect_phrase_hits(
            normalized_text,
            GEO_SPATIAL_PHRASES,
        ),
    }


def normalize_role_identity_slots(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        normalized_value = ROLE_IDENTITY_NORMALIZATION.get(str(value).strip(), str(value).strip())
        _append_unique(normalized, normalized_value)
    return normalized


def derive_comparison_axis_slots(
    text: str,
    slot_inventory: Dict[str, List[str]] | None = None,
) -> List[str]:
    normalized_text = _normalize_slot_text(text)
    inventory = slot_inventory or extract_support_slot_inventory(text)
    axes: List[str] = []

    for value in inventory.get("time_lifecycle_slots", []):
        _append_unique(axes, value)

    comparison_values = set(inventory.get("comparison_or_state_slots", []))
    if {"same-country", "same-nationality"} & comparison_values:
        _append_unique(axes, "nationality")
    if {"majority", "minority", "control-gained"} & comparison_values:
        _append_unique(axes, "control")
    if {"largest", "most-populous"} & comparison_values:
        _append_unique(axes, "population")
    if "wettest" in comparison_values:
        _append_unique(axes, "precipitation")
    if {"second-pick", "first-pick"} & comparison_values:
        _append_unique(axes, "draft-order")
    if "most-games" in comparison_values:
        _append_unique(axes, "games-played")
    if "only-larger" in comparison_values:
        _append_unique(axes, "organization-size")
    if "lowest" in comparison_values and "batting average" in normalized_text:
        _append_unique(axes, "batting-average")

    for axis_value in _collect_phrase_hits(normalized_text, COMPARISON_AXIS_PHRASES):
        _append_unique(axes, axis_value)

    return axes


def extract_question_precision_state(question: str) -> Dict[str, List[str]]:
    inventory = extract_support_slot_inventory(question)
    return {
        "role_identity_slots": normalize_role_identity_slots(inventory["role_identity_slots"]),
        "comparison_axis_slots": derive_comparison_axis_slots(question, inventory),
        **inventory,
    }


def extract_fact_precision_state(subject: str, relation: str, obj: str) -> Dict[str, List[str]]:
    fact_text = " ".join(
        value for value in [subject or "", relation or "", obj or ""] if value
    )
    inventory = extract_support_slot_inventory(fact_text)
    return {
        "role_identity_slots": normalize_role_identity_slots(inventory["role_identity_slots"]),
        "comparison_axis_slots": derive_comparison_axis_slots(fact_text, inventory),
        **inventory,
    }


def extract_question_support_slots(question: str) -> Dict[str, List[str]]:
    return extract_support_slot_inventory(question)


def extract_fact_support_slots(subject: str, relation: str, obj: str) -> Dict[str, List[str]]:
    fact_text = " ".join(
        value for value in [subject or "", relation or "", obj or ""] if value
    )
    return extract_support_slot_inventory(fact_text)
