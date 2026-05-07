# LiteViLNet Agent Instructions

- Treat this as an independent research repository, not a mirror of `/home/admin1/Mycode/VLLiNet`.
- Do not copy datasets, logs, old releases, or historical experiment folders into this repo.
- Keep external leaderboard implementations isolated under `third_party/`; connect them through `adapters/`.
- Keep default paths under the new structure: `datasets/data/kitti_road`, `checkpoints/litevillinet`, `experiments/runs`, and `deployment/*`.
- Do not use the archived VLLiNet V6/Acc checkpoint as a mainline result unless matching code is restored and strict checkpoint loading passes.
