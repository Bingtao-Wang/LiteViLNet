"""Model factory for LiteViLNet research and deployment experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from litevilnet.models.vllinet import VLLiNet_Lite
from litevilnet.models.vllinet_ablation import get_ablation_model


MODEL_PRESETS = {
    "vllinet_paper": {
        "label": "VLLiNet-Paper",
        "builder": "vllinet_lite",
        "checkpoint_hint": "weights/seed/vllinet_paper_v3_final.pth",
        "expected_maxf": 96.35,
        "role": "accuracy reference",
    },
    "vllinet_edge": {
        "label": "VLLiNet-Edge",
        "builder": "ablation:add_lidar",
        "checkpoint_hint": "weights/seed/vllinet_edge_add_lidar.pth",
        "expected_maxf": 96.04,
        "role": "lightweight reference",
    },
    "litevillinet_baseline": {
        "label": "LiteViLNet-Baseline",
        "builder": "ablation:add_lidar",
        "checkpoint_hint": "",
        "expected_maxf": None,
        "role": "new model iteration baseline",
    },
}


def available_presets() -> list[str]:
    return sorted(MODEL_PRESETS)


def create_model(preset: str = "litevillinet_baseline", pretrained: bool = False, deep_supervision: bool = False) -> torch.nn.Module:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {available_presets()}")

    builder = MODEL_PRESETS[preset]["builder"]
    if builder == "vllinet_lite":
        return VLLiNet_Lite(pretrained=pretrained, use_deep_supervision=deep_supervision)
    if builder.startswith("ablation:"):
        config = builder.split(":", 1)[1]
        return get_ablation_model(config, pretrained=pretrained)
    raise ValueError(f"Unsupported builder: {builder}")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "ema_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def load_weights(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = True) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    result = model.load_state_dict(state_dict, strict=strict)
    return {
        "path": str(checkpoint_path),
        "strict": strict,
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "best_metric": checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None,
    }


def assert_compatible_load(load_info: dict[str, Any], allow_aux_heads: bool = True) -> None:
    def is_allowed_aux(key: str) -> bool:
        return allow_aux_heads and key.startswith("decoder.aux_heads.")

    bad_missing = [key for key in load_info.get("missing_keys", []) if not is_allowed_aux(key)]
    bad_unexpected = [key for key in load_info.get("unexpected_keys", []) if not is_allowed_aux(key)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint architecture does not match the requested preset. "
            f"Missing keys: [{', '.join(bad_missing[:8])}] "
            f"Unexpected keys: [{', '.join(bad_unexpected[:8])}]."
        )


def build_model(
    preset: str,
    checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
    pretrained: bool = False,
    deep_supervision: bool = False,
    strict: bool = True,
    allow_partial: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = create_model(preset=preset, pretrained=pretrained, deep_supervision=deep_supervision)
    metadata: dict[str, Any] = {"preset": preset, **MODEL_PRESETS[preset]}

    if checkpoint:
        try:
            metadata["checkpoint"] = load_weights(model, checkpoint, strict=strict)
        except RuntimeError:
            if strict:
                raise
            metadata["checkpoint"] = load_weights(model, checkpoint, strict=False)
        if not allow_partial:
            assert_compatible_load(metadata["checkpoint"])

    model.to(device)
    model.eval()
    metadata["parameters"] = sum(p.numel() for p in model.parameters())
    metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, metadata
