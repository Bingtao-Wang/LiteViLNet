#!/usr/bin/env python
"""Export VLLiNet presets to ONNX for TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from litevilnet.utils.common import write_json
from litevilnet.model_factory import MODEL_PRESETS, available_presets, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LiteViLNet presets to ONNX")
    parser.add_argument("--preset", default="vllinet_paper", choices=available_presets())
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint path")
    parser.add_argument("--output", default="", help="Output ONNX path")
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=1248)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Export with FP16 dummy inputs/model")
    parser.add_argument("--no_check", action="store_true", help="Skip onnx.checker")
    parser.add_argument("--allow_partial", action="store_true", help="Allow non-aux partial checkpoint loading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or MODEL_PRESETS[args.preset].get("checkpoint_hint", "")
    checkpoint = checkpoint if checkpoint and Path(checkpoint).exists() else None

    model, metadata = build_model(
        preset=args.preset,
        checkpoint=checkpoint,
        device=device,
        pretrained=False,
        deep_supervision=False,
        strict=False,
        allow_partial=args.allow_partial,
    )
    if args.fp16:
        model.half()

    output = Path(args.output or f"runs/onnx/{args.preset}_{args.img_h}x{args.img_w}.onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.fp16 else torch.float32
    rgb = torch.randn(1, 3, args.img_h, args.img_w, device=device, dtype=dtype)
    adi = torch.randn(1, 3, args.img_h, args.img_w, device=device, dtype=dtype)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (rgb, adi),
            output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["rgb", "adi"],
            output_names=["logits"],
            dynamic_axes=None,
        )

    check_status = {"checked": False}
    if not args.no_check:
        try:
            import onnx

            onnx_model = onnx.load(str(output))
            onnx.checker.check_model(onnx_model)
            check_status = {"checked": True, "ok": True}
        except Exception as exc:  # pragma: no cover - depends on optional onnx package
            check_status = {"checked": True, "ok": False, "error": str(exc)}

    manifest = {
        "onnx_path": str(output),
        "preset": args.preset,
        "label": metadata["label"],
        "checkpoint": checkpoint,
        "input_shape": [1, 3, args.img_h, args.img_w],
        "fp16_export": args.fp16,
        "opset": args.opset,
        "parameters": metadata["parameters"],
        "onnx_check": check_status,
    }
    write_json(output.with_suffix(".json"), manifest)
    print(f"Exported ONNX: {output}")
    print(f"Manifest: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
