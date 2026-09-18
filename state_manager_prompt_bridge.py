import hashlib
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
            payload = resolve_payload(
                manager_inputs.get("state_json", ""),
                manager_inputs.get("selected_character_id", ""),
                manager_inputs.get("selected_prompt_id", ""),
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

        effective_text = str(controlled_text)
        if text_inputs.get("text") != effective_text:
            text_inputs["text"] = effective_text
            changed += 1

        digest = hashlib.sha256(effective_text.encode("utf-8")).hexdigest()
        timeline = bool(_TIMELINE_HEADER.search(effective_text))

        for impact_id, impact_node, impact_inputs in impacts_by_source[text_id]:
            impact_changed = False
            for key in ("wildcard_text", "populated_text"):
                if impact_inputs.get(key) != effective_text:
                    impact_inputs[key] = effective_text
                    changed += 1
                    impact_changed = True

            _LOG.info(
                "[State Manager] backend Impact prompt bridge text_node=%s manager=%s impact_node=%s impact_class=%s mode=%r role=%r slot=%r chars=%d timeline=%s digest=%s changed=%s",
                text_id,
                manager_id,
                impact_id,
                str(impact_node.get("class_type", "")),
                impact_inputs.get("mode"),
                str(text_inputs.get("role", "positive")),
                str(text_inputs.get("state_slot", "default")),
                len(effective_text),
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
