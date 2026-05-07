#!/usr/bin/env python
"""Build a TensorRT engine with trtexec."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from litevilnet.utils.common import run_command, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engine for LiteViLNet ONNX")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", default="")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace in MiB")
    parser.add_argument("--calib", default="", help="INT8 calibration cache path")
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra trtexec args")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trtexec = shutil.which(args.trtexec)
    if trtexec is None:
        raise SystemExit("trtexec not found. Install TensorRT or pass --trtexec /path/to/trtexec")

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine or f"deployment/artifacts/engines/{onnx_path.stem}_{args.precision}.engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace}",
        "--useCudaGraph",
        "--separateProfileRun",
    ]
    if args.precision == "fp16":
        command.append("--fp16")
    elif args.precision == "int8":
        command.append("--int8")
        if args.calib:
            command.append(f"--calib={args.calib}")
    command.extend(args.extra)

    return_code, output = run_command(command)
    manifest = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "return_code": return_code,
        "command": command,
        "trtexec_output": output,
    }
    write_json(engine_path.with_suffix(".build.json"), manifest)
    print(output)
    if return_code != 0:
        raise SystemExit(return_code)
    print(f"Saved TensorRT engine: {engine_path}")


if __name__ == "__main__":
    main()

