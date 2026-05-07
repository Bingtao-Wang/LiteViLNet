# Third-Party Model Policy

Place downloaded KITTI Road leaderboard implementations under:

```text
third_party/kitti_leaderboard/<model_name>/
```

Rules:

- Keep upstream code isolated; do not merge it into `litevilnet/`.
- Add `SOURCE.md` for each model with source URL, commit/tag, paper, license, dependencies, and reproduction status.
- Put generated files under the model's `outputs/` directory so Git ignores them.
- Add a thin adapter in `third_party/adapters/` when a third-party model needs to enter LiteViLNet evaluation.
