import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, Optional


_LOG = logging.getLogger(__name__)

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


def materialize_state_manager_impact_prompts(
    json_data: Any,
    *,
    resolve_payload: Callable[[Any, Any, Any, Any], Dict[str, Any]],
    library_user_from_ui_state: Callable[[Any], str],
    text_for_box: Callable[[Optional[Dict[str, Any]], Any, Any], Optional[str]],
) -> int:
    """Materialize authoritative managed text at the backend prompt boundary.

    ImpactWildcardProcessor/Encode execute from populated_text. Their own on-prompt
    handler normally derives populated_text from wildcard_text in populate mode, but
    a managed State Manager Text Box can be represented by a link whose authoritative
    value only exists in the persistent State Manager library.

    Resolve that persistent value server-side and write it into the submitted Impact
    node before execution. The mutation is request-local: it does not rewrite the
    workflow graph, State Manager library, Impact mode, or seed.
    """
    prompt = _prompt_nodes(json_data)
    if not prompt:
        return 0

    impacts_by_source: Dict[str, list] = {}
    for impact_id, impact_node in prompt.items():
        if str(impact_node.get("class_type", "")) not in _IMPACT_CLASSES:
            continue
        impact_inputs = _inputs(impact_node)
        if impact_inputs is None:
            continue
        source_id = _link_source(impact_inputs.get("wildcard_text"))
        if source_id is None:
            # Older frontend queue bridges may already have materialized the
            # STRING input and therefore erased the API-prompt link. Recover the
            # original source from the immutable workflow metadata so the backend
            # remains authoritative across mixed frontend/backend revisions.
            source_id = _workflow_input_source(json_data, impact_id, "wildcard_text")
        if source_id is None:
            continue
        impacts_by_source.setdefault(source_id, []).append((str(impact_id), impact_node, impact_inputs))

    if not impacts_by_source:
        return 0

    changed = 0
    for text_id, text_node in prompt.items():
        text_id = str(text_id)
        if text_id not in impacts_by_source:
            continue
        if str(text_node.get("class_type", "")) not in _TEXT_BOX_CLASSES:
            continue
        text_inputs = _inputs(text_node)
        if text_inputs is None:
            continue

        manager_id = _link_source(text_inputs.get("state_control"))
        if manager_id is None:
            continue
        manager_node = _node(prompt, manager_id)
        manager_inputs = _inputs(manager_node)
        if manager_node is None or manager_inputs is None:
            continue
        if str(manager_node.get("class_type", "")) not in _MANAGER_CLASSES:
            continue

        try:
            ui_state_json = manager_inputs.get("ui_state_json", "")
            library_user_id = library_user_from_ui_state(ui_state_json)
            selected_character_id, selected_prompt_id, selection_source = _queued_runtime_selection(
                ui_state_json,
                manager_inputs.get("selected_character_id", ""),
                manager_inputs.get("selected_prompt_id", ""),
            )
            payload = resolve_payload(
                manager_inputs.get("state_json", ""),
                selected_character_id,
                selected_prompt_id,
                library_user_id,
            )
            controlled_text = text_for_box(
                payload,
                text_inputs.get("role", "positive"),
                text_inputs.get("state_slot", "default"),
            )
        except Exception:
            _LOG.exception(
                "[State Manager] backend prompt bridge could not resolve managed text box node=%s manager=%s",
                text_id,
                manager_id,
            )
            continue

        if controlled_text is None:
            continue

        queued_text = text_inputs.get("text", "")
        queued_chars, queued_timeline, queued_digest = _text_fingerprint(queued_text)
        effective_text = str(controlled_text)
        persistent_chars, timeline, digest = _text_fingerprint(effective_text)
        queued_matches_persistent = str(queued_text) == effective_text

        if not queued_matches_persistent:
            _LOG.warning(
                "[State Manager] managed text queue/persistent mismatch text_node=%s manager=%s selection_source=%s character=%r prompt=%r queued_chars=%d queued_timeline=%s queued_digest=%s persistent_chars=%d persistent_timeline=%s persistent_digest=%s",
                text_id,
                manager_id,
                selection_source,
                selected_character_id,
                selected_prompt_id,
                queued_chars,
                queued_timeline,
                queued_digest,
                persistent_chars,
                timeline,
                digest,
            )

        if text_inputs.get("text") != effective_text:
            text_inputs["text"] = effective_text
            changed += 1

        for impact_id, impact_node, impact_inputs in impacts_by_source[text_id]:
            impact_changed = False
            for key in ("wildcard_text", "populated_text"):
                if impact_inputs.get(key) != effective_text:
                    impact_inputs[key] = effective_text
                    changed += 1
                    impact_changed = True

            _LOG.info(
                "[State Manager] backend Impact prompt bridge text_node=%s manager=%s impact_node=%s impact_class=%s mode=%r role=%r slot=%r selection_source=%s character=%r prompt=%r queued_match=%s chars=%d timeline=%s digest=%s changed=%s",
                text_id,
                manager_id,
                impact_id,
                str(impact_node.get("class_type", "")),
                impact_inputs.get("mode"),
                str(text_inputs.get("role", "positive")),
                str(text_inputs.get("state_slot", "default")),
                selection_source,
                selected_character_id,
                selected_prompt_id,
                queued_matches_persistent,
                persistent_chars,
                timeline,
                digest,
                impact_changed,
            )

    return changed


def register_prompt_bridge(
    PromptServer: Any,
    *,
    resolve_payload: Callable[[Any, Any, Any, Any], Dict[str, Any]],
    library_user_from_ui_state: Callable[[Any], str],
    text_for_box: Callable[[Optional[Dict[str, Any]], Any, Any], Optional[str]],
) -> None:
    server = getattr(PromptServer, "instance", None)
    if server is None or not hasattr(server, "add_on_prompt_handler"):
        raise RuntimeError("ComfyUI PromptServer does not expose add_on_prompt_handler")

    marker = "_dora_state_manager_backend_prompt_bridge_v1"
    if getattr(server, marker, False):
        return

    def on_prompt(json_data: Any):
        try:
            materialize_state_manager_impact_prompts(
                json_data,
                resolve_payload=resolve_payload,
                library_user_from_ui_state=library_user_from_ui_state,
                text_for_box=text_for_box,
            )
        except Exception:
            # Prompt handlers must never make unrelated workflows unqueueable.
            _LOG.exception("[State Manager] backend prompt bridge failed; leaving submitted prompt unchanged.")
        return json_data

    server.add_on_prompt_handler(on_prompt)
    setattr(server, marker, True)
