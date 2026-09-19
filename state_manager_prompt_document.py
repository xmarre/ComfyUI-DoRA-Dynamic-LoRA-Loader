"""Persistent prompt-document contract for State Manager text boxes."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional


PROMPT_DOCUMENT_SCHEMA_VERSION = 1
PROMPT_DOCUMENT_FORMATS = {"inherit", "fixed", "list", "timeline"}
PROMPT_DOCUMENT_ROUTINGS = {"logical_chunks", "physical_timeline"}


def _decimal_string(value: Any) -> str:
    text = str(value).strip()
    if not text or "e" in text.lower() or text.startswith(("+", "-")):
        raise ValueError("chunk_seconds must be a positive decimal string")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("chunk_seconds must be a positive decimal string") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def normalize_prompt_document(value: Any, *, preserve_future: bool = True) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("prompt_document must be an object")
    schema = value.get("schema_version")
    if type(schema) is not int or schema < 1:
        raise ValueError("prompt_document schema_version is invalid")
    if schema != PROMPT_DOCUMENT_SCHEMA_VERSION:
        if preserve_future:
            return deepcopy(value)
        raise ValueError(f"unsupported prompt_document schema {schema}")

    fmt = str(value.get("format", "") or "").strip().lower()
    if fmt not in PROMPT_DOCUMENT_FORMATS:
        raise ValueError("prompt_document format is invalid")
    out: Dict[str, Any] = {"schema_version": PROMPT_DOCUMENT_SCHEMA_VERSION, "format": fmt}
    routing = value.get("routing")
    geometry = value.get("geometry")
    if fmt != "timeline":
        if routing is not None or geometry is not None:
            raise ValueError("non-Timeline prompt documents cannot carry routing or geometry")
        return out

    routing = str(routing or "").strip()
    if routing not in PROMPT_DOCUMENT_ROUTINGS:
        raise ValueError("Timeline prompt_document routing is invalid")
    out["routing"] = routing
    if routing == "physical_timeline":
        if geometry is not None:
            raise ValueError("physical Timeline prompt documents cannot carry logical geometry")
        return out

    if not isinstance(geometry, dict):
        raise ValueError("logical Timeline prompt documents require geometry")
    chunks = geometry.get("chunks")
    if type(chunks) is not int or chunks < 1 or chunks > 16:
        raise ValueError("logical Timeline geometry chunks must be in 1..16")
    seconds = _decimal_string(geometry.get("chunk_seconds"))
    out["geometry"] = {"chunks": chunks, "chunk_seconds": seconds}
    return out


def prompt_document_from_box(box: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(box, dict) or "prompt_document" not in box:
        return None
    return normalize_prompt_document(box.get("prompt_document"), preserve_future=True)


def find_text_box(prompt: Any, role: Any, slot: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    wanted_role = str(role or "positive").strip() or "positive"
    wanted_slot = str(slot or "default").strip() or "default"
    boxes = prompt.get("text_boxes")
    if not isinstance(boxes, list):
        return None
    for box in boxes:
        if (
            isinstance(box, dict)
            and str(box.get("role", "")) == wanted_role
            and str(box.get("slot", "")) == wanted_slot
        ):
            return box
    return None
