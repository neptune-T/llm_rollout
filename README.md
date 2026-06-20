# LMM Rollout Project Control Folder

This folder contains the project memory, planning docs, environment notes,
deployment scripts, and daily logs for the LMM / VLA rollout scaling project.

The large codebases and datasets stay outside this folder:

- `../basecode/behavior-1k-solution`: BEHAVIOR champion solution code.
- `../benchmark/home-robot`: OVMM / HomeRobot code and data.
- `../rltask/Sana`: Sana / Sol-RL code.
- `../Planning`: previous exploration notes, kept as historical reference only.

Use this folder as the main project management entry point:

- `memory/project_memory.md`: persistent project state.
- `logs/daily/`: daily work logs.
- `docs/current_plan.md`: current research and engineering plan.
- `docs/env_setup.md`: local / remote environment status.
- `docs/code_reading/`: notes for BEHAVIOR champion, Sol-RL, and integration.
- `scripts/env_check/`: local smoke checks.
- `scripts/deploy/`: deployment and remote preflight helpers.
- `configs/deploy/lmm_deploy.example.env`: deployment config template.

Current priority:

1. Finish OVMM local setup and data verification.
2. Finish local Sana/Sol-RL preflight.
3. Move rendering rollout to an NVIDIA RTX server if local EGL remains blocked.
# llm_rollout
