# LiteViLNet RA-L Deployment Experiments

LiteViLNet should be evaluated as an accuracy-latency-energy trade-off model for embedded road segmentation.

## Main Lines

- `vllinet_paper`: accuracy reference inherited from VLLiNet.
- `vllinet_edge`: lightweight reference inherited from VLLiNet.
- `litevillinet_baseline`: first LiteViLNet candidate.

## Required Metrics

- KITTI Road MaxF, Precision, Recall, IoU.
- PyTorch FP16 latency and FPS.
- TensorRT FP16 latency and FPS.
- GPU memory, engine size, power, and energy per frame on Jetson Orin NX.

## Acceptance Bar

LiteViLNet candidates should keep MaxF close to the Paper reference while improving at least one deployment axis over `vllinet_edge`: latency, FPS, memory, engine size, or energy per frame.
