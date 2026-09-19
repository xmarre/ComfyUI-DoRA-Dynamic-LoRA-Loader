import copy
import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, Optional


_LOG = logging.getLogger(__name__)

_PROMPT_BRIDGE_ORDERING_VERIFIED = False

_MANAGER_CLASSES = {"State Manager", "DoRA State Manager", "StateManager"}
_TEXT_BOX_CLASSES = {"State Manager Text Box", "StateManagerTextBox"}
_IMPACT_CLASSES = {"ImpactWildcardProcessor", "ImpactWildcardEncode"}
_TIMELINE_HEADER = re.compile(r"(?m)^\s*\[[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?s\]\s*$")


def _prompt_nodes(json_data: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(json_data, dict):
        return {}
    prompt = json_data.get("prompt")
    return prompt if isinstance(prompt, dict) else {}


def _node(prompt: Dict[str, Dict[str, Any]], node_id: Any) -> Optional[Dict[str, Any]]:
    value = prompt.get(str(node_id))
    return value if isinstance(value, dict) else None


def _inputs(node: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(node, dict):
        return None
    value = node.get("inputs")
    return value if isinstance(value, dict) else None


def _link_source(value: Any) -> Optional[str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    source = value[0]
    if source is None:
        return None
    return str(source)


def _queued_runtime_selection(
    ui_state_json: Any,
    fallback_character_id: Any,
    fallback_prompt_id: Any,
) -> tuple[str, str, str]:
    """Return the exact request-local selection when the frontend supplied it."""
    parsed: Dict[str, Any] = {}
    if isinstance(ui_state_json, dict):
        parsed = ui_state_json
    elif isinstance(ui_state_json, str) and ui_state_json.strip():
        try:
            value = json.loads(ui_state_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
        if isinstance(value, dict):
            parsed = value

    queued_character = str(parsed.get("__dsm_queued_runtime_character_id", "") or "")
    queued_prompt = str(parsed.get("__dsm_queued_runtime_prompt_id", "") or "")
    if queued_character and queued_prompt:
        return queued_character, queued_prompt, "queue_metadata"
    return str(fallback_character_id or ""), str(fallback_prompt_id or ""), "manager_inputs"


def _queued_frontend_contract(ui_state_json: Any) -> tuple[int, str]:
    parsed: Dict[str, Any] = {}
    if isinstance(ui_state_json, dict):
        parsed = ui_state_json
    elif isinstance(ui_state_json, str) and ui_state_json.strip():
        try:
            value = json.loads(ui_state_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
        if isinstance(value, dict):
            parsed = value
    try:
        version = int(parsed.get("__dsm_frontend_prompt_contract_version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    revision = str(parsed.get("__dsm_frontend_prompt_contract_revision", "") or "")
    return version, revision


def _text_fingerprint(value: Any) -> tuple[int, bool, str]:
    text = str(value or "")
    return (
        len(text),
        bool(_TIMELINE_HEADER.search(text)),
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _workflow(json_data: Any) -> Dict[str, Any]:
    if not isinstance(json_data, dict):
        return {}
    extra = json_data.get("extra_data")
    if not isinstance(extra, dict):
        return {}
    pnginfo = extra.get("extra_pnginfo")
    if not isinstance(pnginfo, dict):
        return {}
    workflow = pnginfo.get("workflow")
    return workflow if isinstance(workflow, dict) else {}


def _workflow_input_source(json_data: Any, node_id: Any, input_name: str) -> Optional[str]:
    workflow = _workflow(json_data)
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        return None

    workflow_node = next(
        (
            item
            for item in nodes
            if isinstance(item, dict) and str(item.get("id")) == str(node_id)
        ),
        None,
    )
    if not isinstance(workflow_node, dict):
        return None

    inputs = workflow_node.get("inputs")
    if not isinstance(inputs, list):
        return None
    input_entry = next(
        (
            item
            for item in inputs
            if isinstance(item, dict) and str(item.get("name", "")) == str(input_name)
        ),
        None,
    )
    if not isinstance(input_entry, dict):
        return None

    link_id = input_entry.get("link")
    if link_id is None:
        return None
    for row in links:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        if str(row[0]) != str(link_id):
            continue
        if str(row[3]) != str(node_id):
            continue
        return str(row[1])
    return None


def _link(value: Any) -> tuple[Optional[str], Optional[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    try:
        output = int(value[1])
    except (TypeError, ValueError):
        return None, None
    return str(value[0]), output


def _box_from_payload(payload: Any, role: Any, slot: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    wanted_role = str(role or "positive")
    wanted_slot = str(slot or "default")
    boxes = payload.get("text_boxes")
    if not isinstance(boxes, list):
        return None
    for box in boxes:
        if (
            isinstance(box, dict)
            and str(box.get("role", "")) == wanted_role
            and str(box.get("slot", "default")) == wanted_slot
        ):
            return box
    return None


def _parse_ui_state(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _discard_queue_snapshot(manager_inputs: Dict[str, Any]) -> bool:
    """Remove an untrusted client-supplied reserved queue snapshot."""
    ui_state = _parse_ui_state(manager_inputs.get("ui_state_json", ""))
    if "__dsm_queue_snapshot_v1" not in ui_state:
        return False
    ui_state.pop("__dsm_queue_snapshot_v1", None)
    manager_inputs["ui_state_json"] = json.dumps(
        ui_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return True


def _install_queue_snapshot(manager_inputs: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
    ui_state = _parse_ui_state(manager_inputs.get("ui_state_json", ""))
    # The reserved snapshot is server-owned. Never trust a client copy.
    ui_state.pop("__dsm_queue_snapshot_v1", None)
    ui_state["__dsm_queue_snapshot_v1"] = {
        "version": 1,
        "library_revision": int(snapshot["library_revision"]),
        "character_id": str(snapshot["character_id"]),
        "prompt_id": str(snapshot["prompt_id"]),
        "payload": snapshot["payload"],
    }
    manager_inputs["ui_state_json"] = json.dumps(
        ui_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _continuum_provider(class_type: Any) -> Optional[Dict[str, Any]]:
    """Discover the public provider only through ComfyUI's registered node classes."""
    try:
        import nodes as comfy_nodes

        cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(str(class_type))
        provider = getattr(cls, "H3_CONTINUUM_PROMPT_TRANSPORT_PROVIDER_V1", None)
        if not isinstance(provider, dict) or provider.get("provider_version") != 1:
            return None
        if not callable(provider.get("classify")) or not callable(provider.get("inspect")):
            return None
        input_types = cls.INPUT_TYPES()
        inputs = {}
        for section in ("required", "optional"):
            section_inputs = input_types.get(section)
            if isinstance(section_inputs, dict):
                inputs.update(section_inputs)
        if "sequence_prompt" not in inputs or "managed_prompt_source_json" not in inputs:
            return None
        chunks = provider.get("chunks")
        seconds = provider.get("chunk_seconds")
        if not isinstance(chunks, dict) or not isinstance(seconds, dict):
            return None
        return provider
    except Exception:
        return None


def prompt_transport_ordering_contract() -> Optional[str]:
    return "ordered-impact-v1" if _PROMPT_BRIDGE_ORDERING_VERIFIED else None


def prompt_transport_provider_capabilities() -> Optional[Dict[str, Any]]:
    """Return JSON-safe public capability data for the State Manager editor."""
    try:
        import nodes as comfy_nodes

        for cls in getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).values():
            provider = getattr(cls, "H3_CONTINUUM_PROMPT_TRANSPORT_PROVIDER_V1", None)
            if not isinstance(provider, dict) or provider.get("provider_version") != 1:
                continue
            if not callable(provider.get("classify")) or not callable(provider.get("inspect")):
                continue
            return {
                key: value
                for key, value in provider.items()
                if key not in {"classify", "inspect"} and isinstance(value, (str, int, float, bool, list, dict, type(None)))
            }
    except Exception:
        return None
    return None


def _document_supported_by_provider(document: Any, provider: Dict[str, Any]) -> bool:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return False
    if document.get("format") != "timeline" or document.get("routing") != "logical_chunks":
        return True
    geometry = document.get("geometry")
    if not isinstance(geometry, dict):
        return False
    try:
        chunks = int(geometry.get("chunks"))
        seconds = float(geometry.get("chunk_seconds"))
        chunk_range = provider["chunks"]
        seconds_range = provider["chunk_seconds"]
        return (
            int(chunk_range["min"]) <= chunks <= int(chunk_range["max"])
            and float(seconds_range["min"]) <= seconds <= float(seconds_range["max"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _sidecar(
    *,
    text: str,
    prompt_document: Dict[str, Any],
    snapshot: Dict[str, Any],
    manager_id: str,
    text_id: str,
    impact_id: Optional[str],
    role: str,
    slot: str,
) -> str:
    return json.dumps(
        {
            "magic": "DSM_H3_PROMPT_SOURCE",
            "schema_version": 1,
            "text": text,
            "prompt_document": prompt_document,
            "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "library_revision": int(snapshot["library_revision"]),
            "binding": {
                "manager_node": manager_id,
                "text_node": text_id,
                "impact_node": impact_id,
                "role": role,
                "slot": slot,
            },
            "queue_contract": "ordered-impact-v1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def materialize_state_manager_impact_prompts(
    json_data: Any,
    *,
    resolve_payload: Callable[[Any, Any, Any, Any], Dict[str, Any]],
    library_user_from_ui_state: Callable[[Any], str],
    text_for_box: Callable[[Optional[Dict[str, Any]], Any, Any], Optional[str]],
    resolve_snapshot: Optional[Callable[[Any, Any, Any, Any], Dict[str, Any]]] = None,
    ordering_verified: bool = True,
) -> int:
    """Freeze managed state once, materialize Impact input, and attach verified sidecars."""
    prompt = _prompt_nodes(json_data)
    if not prompt:
        return 0

    changed = 0
    snapshots: Dict[str, Dict[str, Any]] = {}
    manager_context: Dict[str, Dict[str, Any]] = {}
    managed: Dict[str, Dict[str, Any]] = {}

    # Freeze every queued State Manager exactly once, independently of whether a
    # managed Text Box is present. The reserved UI-state field is server-owned:
    # discard any submitted copy before resolving the authoritative snapshot.
    for manager_id, manager_node in prompt.items():
        manager_id = str(manager_id)
        if str(manager_node.get("class_type", "")) not in _MANAGER_CLASSES:
            continue
        manager_inputs = _inputs(manager_node)
        if manager_inputs is None:
            continue
        if _discard_queue_snapshot(manager_inputs):
            changed += 1
        ui_state_json = manager_inputs.get("ui_state_json", "")
        library_user_id = library_user_from_ui_state(ui_state_json)
        selected_character_id, selected_prompt_id, selection_source = _queued_runtime_selection(
            ui_state_json,
            manager_inputs.get("selected_character_id", ""),
            manager_inputs.get("selected_prompt_id", ""),
        )
        manager_context[manager_id] = {
            "inputs": manager_inputs,
            "library_user_id": library_user_id,
            "selected_character_id": selected_character_id,
            "selected_prompt_id": selected_prompt_id,
            "selection_source": selection_source,
        }
        try:
            if resolve_snapshot is not None:
                snapshot = resolve_snapshot(
                    manager_inputs.get("state_json", ""),
                    selected_character_id,
                    selected_prompt_id,
                    library_user_id,
                )
            else:
                payload = resolve_payload(
                    manager_inputs.get("state_json", ""),
                    selected_character_id,
                    selected_prompt_id,
                    library_user_id,
                )
                snapshot = {
                    "version": 1,
                    "library_revision": -1,
                    "character_id": selected_character_id,
                    "prompt_id": selected_prompt_id,
                    "payload": payload,
                }
            if not isinstance(snapshot, dict) or not isinstance(snapshot.get("payload"), dict):
                raise ValueError("invalid State Manager queue snapshot")
            snapshots[manager_id] = snapshot
            if int(snapshot.get("library_revision", -1)) >= 0:
                _install_queue_snapshot(manager_inputs, snapshot)
                changed += 1
        except Exception:
            _LOG.exception(
                "[State Manager] backend prompt bridge could not freeze manager=%s",
                manager_id,
            )

    # Materialize each managed Text Box from its owning manager's frozen payload.
    # This makes fan-out deterministic and keeps text/settings/references on one
    # library revision for the complete queued request.
    for text_id, text_node in prompt.items():
        text_id = str(text_id)
        if str(text_node.get("class_type", "")) not in _TEXT_BOX_CLASSES:
            continue
        text_inputs = _inputs(text_node)
        if text_inputs is None:
            continue
        manager_id = _link_source(text_inputs.get("state_control"))
        if manager_id is None:
            continue
        context = manager_context.get(manager_id)
        snapshot = snapshots.get(manager_id)
        if context is None or snapshot is None:
            continue

        payload = snapshot["payload"]
        role = str(text_inputs.get("role", "positive"))
        slot = str(text_inputs.get("state_slot", "default"))
        controlled_text = text_for_box(payload, role, slot)
        if controlled_text is None:
            continue
        effective_text = str(controlled_text)
        if text_inputs.get("text") != effective_text:
            text_inputs["text"] = effective_text
            changed += 1
        box = _box_from_payload(payload, role, slot)
        managed[text_id] = {
            "manager_id": manager_id,
            "snapshot": snapshot,
            "text": effective_text,
            "prompt_document": box.get("prompt_document") if isinstance(box, dict) else None,
            "role": role,
            "slot": slot,
            "selection_source": context["selection_source"],
        }

    # Preserve native Impact semantics: both source widgets are made authoritative,
    # but mode and seed remain untouched for Impact's own handler/execution.
    impact_sources: Dict[str, Dict[str, Any]] = {}
    for impact_id, impact_node in prompt.items():
        impact_id = str(impact_id)
        if str(impact_node.get("class_type", "")) not in _IMPACT_CLASSES:
            continue
        impact_inputs = _inputs(impact_node)
        if impact_inputs is None:
            continue
        source_id, output_index = _link(impact_inputs.get("wildcard_text"))
        if source_id is None:
            source_id = _workflow_input_source(json_data, impact_id, "wildcard_text")
            output_index = 0 if source_id is not None else None
        if output_index not in (None, 0) or source_id not in managed:
            continue
        info = managed[source_id]
        effective_text = info["text"]
        impact_changed = False
        for key in ("wildcard_text", "populated_text"):
            if impact_inputs.get(key) != effective_text:
                impact_inputs[key] = effective_text
                changed += 1
                impact_changed = True
        impact_sources[impact_id] = {**info, "text_id": source_id}

        frontend_contract_version, frontend_contract_revision = _queued_frontend_contract(
            _inputs(_node(prompt, info["manager_id"])).get("ui_state_json", "")
        )
        chars, timeline, digest = _text_fingerprint(effective_text)
        _LOG.info(
            "[State Manager] backend Impact prompt bridge revision=identity-v2 contract=v5 "
            "frontend_contract=%d frontend_revision=%r text_node=%s manager=%s impact_node=%s "
            "impact_class=%s mode=%r role=%r slot=%r selection_source=%s chars=%d timeline=%s digest=%s changed=%s",
            frontend_contract_version,
            frontend_contract_revision,
            source_id,
            info["manager_id"],
            impact_id,
            str(impact_node.get("class_type", "")),
            impact_inputs.get("mode"),
            info["role"],
            info["slot"],
            info["selection_source"],
            chars,
            timeline,
            digest,
            impact_changed,
        )

    if not ordering_verified:
        return changed

    # Attach sidecars only to exact supported sequence_prompt wires. Unknown
    # transforms, non-zero outputs, cycles, or consumers without provider v1
    # deliberately stay on legacy STRING execution.
    for consumer_id, consumer_node in prompt.items():
        consumer_inputs = _inputs(consumer_node)
        if consumer_inputs is None:
            continue
        provider = _continuum_provider(consumer_node.get("class_type"))
        if provider is None:
            continue
        source_id, output_index = _link(consumer_inputs.get("sequence_prompt"))
        if source_id is None or output_index != 0:
            continue

        impact_id: Optional[str] = None
        if source_id in managed:
            info = managed[source_id]
            text_id = source_id
        elif source_id in impact_sources:
            info = impact_sources[source_id]
            impact_id = source_id
            # The Impact inputs were intentionally materialized above. Use the
            # source id captured before that replacement instead of attempting
            # to rediscover a link that no longer exists in the prompt payload.
            text_id = str(info.get("text_id", "") or "")
            if text_id not in managed:
                continue
        else:
            continue

        document = info.get("prompt_document")
        if not _document_supported_by_provider(document, provider):
            _LOG.warning(
                "[State Manager] managed prompt sidecar skipped consumer=%s source=%s reason=unsupported_descriptor",
                consumer_id,
                source_id,
            )
            continue
        snapshot = info["snapshot"]
        if int(snapshot.get("library_revision", -1)) < 0:
            continue
        sidecar = _sidecar(
            text=info["text"],
            prompt_document=document,
            snapshot=snapshot,
            manager_id=info["manager_id"],
            text_id=str(text_id),
            impact_id=impact_id,
            role=info["role"],
            slot=info["slot"],
        )
        if consumer_inputs.get("managed_prompt_source_json") != sidecar:
            consumer_inputs["managed_prompt_source_json"] = sidecar
            changed += 1

    return changed


def _handler_order_receipt(handlers: Any, *, limit: int = 32) -> Dict[str, Any]:
    values = list(handlers) if isinstance(handlers, list) else []
    shown = []
    for handler in values[: max(0, int(limit))]:
        shown.append(
            {
                "module": str(getattr(handler, "__module__", "") or ""),
                "name": str(
                    getattr(handler, "__qualname__", None)
                    or getattr(handler, "__name__", None)
                    or type(handler).__name__
                ),
            }
        )
    return {
        "count": len(values),
        "shown": shown,
        "truncated": len(values) > len(shown),
    }


def register_prompt_bridge(
    PromptServer: Any,
    *,
    resolve_payload: Callable[[Any, Any, Any, Any], Dict[str, Any]],
    library_user_from_ui_state: Callable[[Any], str],
    text_for_box: Callable[[Optional[Dict[str, Any]], Any, Any], Optional[str]],
    resolve_snapshot: Optional[Callable[[Any, Any, Any, Any], Dict[str, Any]]] = None,
) -> None:
    server = getattr(PromptServer, "instance", None)
    if server is None or not hasattr(server, "add_on_prompt_handler"):
        raise RuntimeError("ComfyUI PromptServer does not expose add_on_prompt_handler")

    global _PROMPT_BRIDGE_ORDERING_VERIFIED

    marker = "_dora_state_manager_backend_prompt_bridge_callback_v2"
    handlers = getattr(server, "on_prompt_handlers", None)
    ordering_verified = isinstance(handlers, list)
    _PROMPT_BRIDGE_ORDERING_VERIFIED = ordering_verified

    def on_prompt(json_data: Any):
        # PromptServer JSON is request-local plain data. Stage every manager,
        # Impact and consumer mutation on a detached copy so a late resolution
        # failure cannot leave only part of a fan-out materialized.
        try:
            working = copy.deepcopy(json_data)
            materialize_state_manager_impact_prompts(
                working,
                resolve_payload=resolve_payload,
                library_user_from_ui_state=library_user_from_ui_state,
                text_for_box=text_for_box,
                resolve_snapshot=resolve_snapshot,
                ordering_verified=ordering_verified,
            )
            return working
        except Exception:
            _LOG.exception("[State Manager] backend prompt bridge failed; leaving submitted prompt unchanged.")
            return json_data

    previous = getattr(server, marker, None)
    if ordering_verified:
        if callable(previous) and previous in handlers:
            handlers.remove(previous)
        handlers.insert(0, on_prompt)
        setattr(server, marker, on_prompt)
    else:
        # Compatibility fallback retains v4 materialization but cannot advertise
        # ordered-impact-v1 or attach managed sequence sidecars. Without access to
        # the handler list we cannot safely remove a previous callback, so keep
        # the already-registered owned callback rather than double-registering.
        if not callable(previous):
            server.add_on_prompt_handler(on_prompt)
            setattr(server, marker, on_prompt)
    handler_order = (
        _handler_order_receipt(handlers)
        if ordering_verified
        else {"count": None, "shown": [], "truncated": False}
    )
    _LOG.info(
        "[State Manager] backend prompt bridge registered revision=identity-v2 contract=v5 "
        "backend_write=backend-document-write-v1 ordering_verified=%s handler_order=%s",
        ordering_verified,
        json.dumps(handler_order, sort_keys=True, separators=(",", ":")),
    )
