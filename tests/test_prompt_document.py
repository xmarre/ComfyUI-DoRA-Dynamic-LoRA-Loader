from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


ROOT = Path(os.environ.get("DORA_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_manager_prompt_document import normalize_prompt_document


def test_logical_timeline_geometry_normalizes_trailing_zeroes():
    assert normalize_prompt_document({
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5.000"},
    }) == {
        "schema_version": 1,
        "format": "timeline",
        "routing": "logical_chunks",
        "geometry": {"chunks": 3, "chunk_seconds": "5"},
    }


@pytest.mark.parametrize("seconds", [5, 5.0, ".5", "05", "1.", "+5", "-5", "5e0", "", "nan", "inf"])
def test_logical_timeline_geometry_rejects_noncanonical_decimal_spellings(seconds):
    with pytest.raises(ValueError, match="chunk_seconds"):
        normalize_prompt_document({
            "schema_version": 1,
            "format": "timeline",
            "routing": "logical_chunks",
            "geometry": {"chunks": 1, "chunk_seconds": seconds},
        })


def test_future_prompt_document_is_preserved_opaquely_only_when_requested():
    future = {
        "schema_version": 99,
        "future_mode": "opaque",
        "nested": {"keep": [1, 2, 3]},
    }
    assert normalize_prompt_document(future, preserve_future=True) == future
    with pytest.raises(ValueError, match="unsupported prompt_document schema 99"):
        normalize_prompt_document(future, preserve_future=False)


@pytest.mark.parametrize("fmt", ["inherit", "fixed", "list"])
def test_non_timeline_document_rejects_routing_or_geometry(fmt):
    with pytest.raises(ValueError, match="cannot carry routing or geometry"):
        normalize_prompt_document({
            "schema_version": 1,
            "format": fmt,
            "routing": "logical_chunks",
        })


def test_physical_timeline_rejects_logical_geometry():
    with pytest.raises(ValueError, match="cannot carry logical geometry"):
        normalize_prompt_document({
            "schema_version": 1,
            "format": "timeline",
            "routing": "physical_timeline",
            "geometry": {"chunks": 1, "chunk_seconds": "5"},
        })
