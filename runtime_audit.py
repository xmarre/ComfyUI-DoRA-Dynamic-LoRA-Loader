"""Opt-in, bounded receipts for the user adapter's actual forward execution."""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import hashlib
import json
import logging
import re

import torch
import torch.nn.functional as F


_LOG = logging.getLogger(__name__)


def target_family(key):
    name = key.lower()
    for suffix, family in (("qkv_proj.weight", "QKV"), ("q_proj.weight", "Q"),
                           ("k_proj.weight", "K"), ("v_proj.weight", "V"),
                           ("out_proj.weight", "O")):
        if name.endswith(suffix):
            return family
    if ".mlp." in name or ".ffn." in name:
        return "MLP"
    if "adaln" in name or "modulation" in name:
        return "modulation"
    return "other"


def _stats(value):
    value = value.detach().float().cpu().contiguous()
    return {"sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "rms": float(value.square().mean().sqrt()), "mean": float(value.mean())}


def _sample_rows(start, stop):
    return sorted({start, (start + stop - 1) // 2, stop - 1}) if stop > start else []


class RuntimeAudit:
    """One loader-owned observer; context and counts live for one model call."""

    def __init__(self, provider):
        self.provider = provider
        self.entries = []
        self._sampled_stages = set()
        self._call = ContextVar(f"dora_audit_{id(self)}", default=None)

    def emit(self, kind, **fields):
        record = {"schema": "dora_runtime_audit_v1", "provider": self.provider, **fields}
        _LOG.info("[DoRA runtime audit] %s %s", kind, json.dumps(record, sort_keys=True))
        call = self._call.get()
        metrics = call.get("metrics") if call else None
        if metrics is not None and callable(getattr(metrics, "event", None)):
            metrics.event(kind, **record)

    def attach(self, hook, item):
        index = len(self.entries)
        weight = hook.module.weight
        self.entries.append({"index": index, "key": item["key"],
                             "family": target_family(item["key"]),
                             "adapter": type(hook.adapter).__name__,
                             "strength": hook.multiplier, "module_id": id(hook.module),
                             "hook_id": id(hook), "weight_layout": getattr(weight, "_layout_cls", None)})
        original_h = hook.adapter.h

        def observed_h(x, base_out):
            residual = original_h(x, base_out)
            self.observe(index, x, base_out, residual, hook.adapter)
            return residual

        hook.adapter.h = observed_h

    def manifest(self):
        self._sampled_stages.clear()
        self.emit("user_lora_manifest", targets=self.entries,
                  family_counts=dict(Counter(e["family"] for e in self.entries)))

    def wrapper(self, executor, *args, **kwargs):
        options = kwargs.get("transformer_options", args[3] if len(args) > 3 else {}) or {}
        runtime = options.get("h3_flow_partitioned_stage_v1")
        plan = getattr(runtime, "plan", None)
        payload = kwargs.get("minimax_payload") or {}
        stage = options.get("h3_flow_stage", "native")
        call = {"counts": [0] * len(self.entries), "sampled": set(), "plan": plan,
                "sample_residual": stage not in self._sampled_stages,
                "layout": payload.get("layout"), "metrics": getattr(runtime, "metrics", None),
                "stage": stage,
                "request_id": options.get("h3_flow_request_id_v1"),
                "evaluation_id": options.get("h3_flow_evaluation_id_v1")}
        token = self._call.set(call)
        complete = False
        try:
            result = executor(*args, **kwargs)
            complete = True
            self._sampled_stages.add(stage)
            return result
        finally:
            try:
                missing = [e["key"] for e, count in zip(self.entries, call["counts"]) if count == 0]
                self.emit("user_lora_execution", stage=call["stage"],
                          request_id=call["request_id"], evaluation_id=call["evaluation_id"],
                          completed=complete, counts_by_manifest_index=call["counts"],
                          missing_targets=missing)
            finally:
                self._call.reset(token)

    def _regions(self, call, rows):
        layout, plan = call["layout"], call["plan"]
        regions = []
        segments = getattr(layout, "segments", ())
        if plan is not None:
            video_start = rows - plan.partitioned_rows
            if video_start < 0:
                return [("unmapped", 0, rows)]
            regions.extend((kind, start, stop) for start, stop, kind in segments
                           if kind != "video" and 0 <= start < stop <= video_start)
            prefix_stop = video_start + plan.prefix_rows
            boundary_stop = min(rows, prefix_stop + 3 * plan.source_rows)
            regions.extend((("carried_prefix_video", video_start, prefix_stop),
                            ("first_generated_group", prefix_stop, boundary_stop),
                            ("later_suffix", boundary_stop, rows)))
            for frame in range(max(0, plan.prefix_t - 4), min(plan.temporal, plan.prefix_t + 5)):
                start = (video_start + frame * plan.target_rows if frame < plan.prefix_t
                         else prefix_stop + (frame - plan.prefix_t) * plan.source_rows)
                stop = start + (plan.target_rows if frame < plan.prefix_t else plan.source_rows)
                regions.append((f"video_frame_{frame}", start, stop))
        elif segments and getattr(layout, "seq_len", None) == rows:
            regions.extend((kind, start, stop) for start, stop, kind in segments)
        else:
            regions.append(("unmapped", 0, rows))
        return regions

    def observe(self, index, x, base, residual, adapter):
        call = self._call.get()
        if call is None:
            return
        call["counts"][index] += 1
        entry = self.entries[index]
        # First actual NFE per stage/injection; Spectrum forecasts never call us.
        if (not call["sample_residual"] or index in call["sampled"] or "token_refiner" in entry["key"]
                or not re.search(r"\.blocks\.(0|24|49)\.", entry["key"])
                or x.ndim not in (2, 3) or base.shape != residual.shape):
            return
        call["sampled"].add(index)
        x_rows = x.reshape(-1, x.shape[-2], x.shape[-1])[0]
        base_rows = base.reshape(-1, base.shape[-2], base.shape[-1])[0]
        residual_rows = residual.reshape(-1, residual.shape[-2], residual.shape[-1])[0]
        width = base_rows.shape[-1]
        projections = [(entry["family"], 0, width)]
        if entry["family"] == "QKV" and width % 3 == 0:
            projections = [(name, i * (width // 3), (i + 1) * (width // 3))
                           for i, name in enumerate(("Q", "K", "V"))]
        receipts = []
        # Only three rows and up to sixteen output channels per region/projection.
        with torch.no_grad():
            for domain, start, stop in self._regions(call, base_rows.shape[-2]):
                indices = _sample_rows(start, stop)
                if not indices:
                    continue
                input_sample = x_rows[indices].detach().float()
                input_stats = _stats(input_sample)
                for projection, first, last in projections:
                    channels = torch.linspace(first, last - 1, min(16, last - first),
                                              device=base.device).long()
                    b = base_rows[indices][:, channels].detach().float()
                    d = residual_rows[indices][:, channels].detach().float()
                    b_stats, d_stats = _stats(b), _stats(d)
                    receipt = {"domain": domain, "projection": projection, "rows": indices,
                               "channels": channels.cpu().tolist(), "input": input_stats,
                               "base": b_stats, "residual": d_stats,
                               "post": _stats(base_rows[indices][:, channels] + residual_rows[indices][:, channels]),
                               "residual_base_rms_ratio": d_stats["rms"] / max(b_stats["rms"], 1e-30)}
                    weights = getattr(adapter, "weights", ())
                    if (type(adapter).__name__ == "LoRAAdapter" and len(weights) == 6
                            and all(value is None for value in weights[3:])):
                        up, down, alpha = weights[:3]
                        scale = adapter.multiplier * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
                        delta_weight = (up[channels].float() @ down.float()) * scale
                        materialized = F.linear(input_sample, delta_weight)
                        error = d - materialized
                        receipt["materialized_residual_error_rms"] = float(error.square().mean().sqrt())
                        receipt["materialized_residual_error_max"] = float(error.abs().max())
                    receipts.append(receipt)
        self.emit("user_lora_residual", stage=call["stage"], request_id=call["request_id"],
                  evaluation_id=call["evaluation_id"], **entry, samples=receipts,
                  sample_semantics="three_rows_up_to_16_channels_batch_0_fp32_digest",
                  base_semantics="before_this_user_adapter; VDN_post_hook_runs_later")
