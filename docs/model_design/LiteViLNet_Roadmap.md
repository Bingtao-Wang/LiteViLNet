# LiteViLNet Roadmap

## Baselines

- `litevilnet_paper`: accuracy reference.
- `litevilnet_edge`: lightweight reference.
- `litevilnet_baseline`: first new-model iteration, initially sharing the lightweight seed architecture.

## Iteration Priorities

1. Preserve KITTI Road MaxF near the LiteViLNet-Paper reference.
2. Reduce latency, parameters, and TensorRT engine size.
3. Keep architecture export-friendly for ONNX and TensorRT.
4. Compare against external KITTI Road leaderboard models through isolated adapters.

## Do Not Use as Mainline

VLLiNet V6/Acc remains archived until its matching full architecture code is restored and strict checkpoint loading is verified.
