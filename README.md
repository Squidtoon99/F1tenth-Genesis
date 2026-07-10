# F1tenth Genesis

## Training

Solo (1v0) standalone training:

```bash
python standalone_trainer.py --num-envs 512 --total-steps 500000
```

Head-to-head 1v1 racing (overtake a scripted opponent) — see
[docs/training_1v1.md](docs/training_1v1.md):

```bash
python standalone_trainer.py --opponent scripted --device cuda --precision 32
```

## Docs

- [docs/training_1v1.md](docs/training_1v1.md) — training with opponents (1v1).
- [docs/observation_audit.md](docs/observation_audit.md) — observation spec and audit
  (includes the 1v1 opponent-block layout).
