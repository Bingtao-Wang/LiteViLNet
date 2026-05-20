"""Download RGB road pseudo-label models into tracked local folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from tools.predict_rgb_pseudolabels import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RGB pseudo-label models")
    parser.add_argument("--models_root", default="other_models")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--endpoint", default="", help="Optional HuggingFace endpoint, e.g. https://hf-mirror.com")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.models_root)
    root.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for name in args.models:
        spec = MODEL_REGISTRY[name]
        model_id = spec["model_id"]
        local_dir = Path(spec.get("local_dir", root / name))
        if not local_dir.is_absolute():
            local_dir = Path.cwd() / local_dir
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {name}: {model_id} -> {local_dir}")
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            force_download=args.force_download,
            endpoint=args.endpoint or None,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite"],
        )
        metadata = {
            "name": name,
            "model_id": model_id,
            "backend": spec["backend"],
            "road_labels": list(spec["road_labels"]),
            "local_dir": str(local_dir),
            "snapshot_path": path,
        }
        (local_dir / "litevilnet_model_manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        downloaded.append(metadata)

    summary = {
        "models_root": str(root),
        "models": downloaded,
    }
    (root / "rgb_pseudolabel_models_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
