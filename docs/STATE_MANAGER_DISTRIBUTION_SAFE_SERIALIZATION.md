# State Manager: distribution-safe workflow serialization

The State Manager keeps its character/prompt library in persistent local storage and normally serializes only workflow binding state plus the currently selected character/prompt UUIDs.

For workflows intended for public distribution, even those local UUID bindings are unnecessary. Enable **Distribution-safe workflow serialization** in the State Manager's **Settings / seed** panel.

When enabled, workflow saves/exports serialize:

```text
selected_character_id = default_character
selected_prompt_id    = default_prompt
```

instead of the currently selected local UUIDs.

## What it does not change

The option only rewrites the serialized workflow output. It does **not** change:

- the live selected character or prompt in the current ComfyUI session;
- the persistent State Manager library;
- connected DoRA/LoRA state;
- prompt/seed/settings state;
- queued runtime selection or generation behavior.

This means a release author can keep working with a private local preset while every workflow save/export remains clean for distribution.

## Workflow-level behavior

The toggle is stored as the workflow node property:

```text
dora_state_manager_distribution_safe_serialization = true
```

It is disabled by default for backward compatibility.

A distributed workflow may intentionally ship with the option enabled. In that case, users who want their own local character/prompt selection to persist into future workflow saves should disable the option after importing the workflow.

## Scope

The feature sanitizes both serialization forms when present:

- positional `widgets_values`;
- named `widgets_values_named`.

It supports both the current `State Manager` node class and the legacy `DoRA State Manager` class.
