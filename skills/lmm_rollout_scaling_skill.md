# Skill: LMM Rollout Scaling

Last updated: 2026-06-20

## What this project is about

This project studies online RL with rollout scaling for long-horizon mobile manipulation. The key question is whether cheap rollout can act as a reliable proxy for high-precision rollout, and how to correct it when ranking collapses under long horizons or contact-rich manipulation.

## What this project is not about

- Not generic LMM as a language model project.
- Not pure VLA quantization.
- Not pure world modeling.
- Not only benchmark score chasing.
- Not planner/controller engineering unless it helps answer rollout scaling reliability and online RL utility.
- Not currently an OpenVLA baseline project.
- Not currently a J-EPA/V-JEPA/world-model reward project.

## Key terminology

- LMM: Long-horizon mobile manipulation.
- VLA: Vision-language-action policy.
- BEHAVIOR-1K: Main candidate benchmark for household mobile manipulation in Isaac Sim / OmniGibson.
- OVMM: First smoke benchmark for easier mobile-manipulation rollout setup in Habitat/HomeRobot.
- BEHAVIOR champion solution: The Pi0.5/OpenPI-derived BEHAVIOR Challenge winning solution used as the engineering baseline.
- Sol-RL: FP4/NVFP4 explore, BF16 train rollout-based RL reference from the Sana repo.
- cheap rollout: Lower-cost candidate rollout mode, such as low precision, smaller model, lower resolution, fewer inference steps, or other fast surrogate.
- high-precision rollout: BF16/FP16/FP32 or otherwise trusted rollout mode used as the reference ranking.
- ranking correlation: Spearman, Kendall tau, top-k overlap, success agreement, false positive and false negative rates between cheap and high-precision candidate rankings.
- rollout scaling: Increasing the number of candidate rollouts to improve policy selection or training signal.
- online RL: Iteratively collecting rollouts from the current policy and updating the policy.
- contact-rich manipulation: Stages where contact, grasp, articulation, placement, and release dominate task success.
- stage-aware verification: Selectively rerunning cheap candidates with high precision based on stage risk, uncertainty, or expected utility.

## Standard workflow

At the beginning of each work session:

1. Read `memory/project_memory.md`.
2. Read the latest three files under `logs/daily/`.
3. Check git status for relevant subrepos: `Planning`, `basecode/behavior-1k-solution`, and `rltask/Sana`.
4. Check each relevant repo's latest commit.
5. Identify the current stage: Stage 0 docs, Stage 1 env/baseline, Stage 2 ranking, Stage 3 horizon/contact, Stage 4 method, or Stage 5 online RL.
6. Check `docs/current_plan.md` before using older notes.
7. State the plan before editing files or running experiments.

Treat `Planning/Doc/Todo.md` as historical exploration context. Do not follow its OpenVLA or J-EPA/V-JEPA route unless the user explicitly changes the project direction.

Current immediate route: OVMM first for the smallest environment loop, then BEHAVIOR champion for the stronger long-horizon platform.

## Deployment workflow

Use the deployment scaffold when moving the workspace to remote machines:

```bash
cp configs/deploy/lmm_deploy.example.env configs/deploy/lmm_deploy.env
bash scripts/deploy/deploy_all.sh configs/deploy/lmm_deploy.env
```

Default deployment syncs code/docs/configs/scripts and excludes checkpoints, datasets, outputs, videos, and virtualenvs. Set `SYNC_CHECKPOINTS=1` only when transferring the 48GB BEHAVIOR checkpoints is intended.

Remote roles:

- NVIDIA render host: BEHAVIOR / OmniGibson / Isaac Sim evaluator and rendering.
- AMD train host: ROCm-compatible training/calibration/analysis after preflight. Access is tunnel-only unless SSH is explicitly enabled.

Do not assume the AMD host can run the champion policy or Sana CUDA setup unchanged.

For AMD tunnel deployment:

```bash
bash scripts/deploy/package_for_tunnel.sh configs/deploy/lmm_deploy.env
```

Upload the tarball in VS Code Tunnel and run:

```bash
cd /scratch/$USER/longhorizon
bash scripts/deploy/remote_preflight_amd_train.sh
```

AMD full-node Slurm allocation:

```bash
salloc --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation --exclusive --mem=0
```

At the end of each work session:

1. Update `logs/daily/YYYY-MM-DD.md`.
2. Update `memory/project_memory.md`.
3. Update this skill file when commands, workflow, or failure diagnostics change.
4. If experiments ran, record full metadata.
5. If code changed, state purpose, files, risk, verification, and proposed commit message.

## How to run OVMM environment check

Local OVMM smoke command:

```bash
bash scripts/env_check/ovmm_local_smoke.sh
```

Known local status:

- `home-robot` conda env exists.
- Host execution sees CUDA from PyTorch.
- OVMM episodes exist under `benchmark/home-robot/data/datasets/ovmm`.
- Local Habitat-Sim currently fails at OpenGL/EGL context creation with `GL::Context: cannot retrieve OpenGL version`.

Sol-RL local preflight:

```bash
bash scripts/env_check/solrl_local_preflight.sh
```

Expected current local status: missing `sana` env. Do not run Sana `environment_setup.sh` locally without explicit confirmation because it installs a heavy CUDA 12.8 training stack.

## How to run BEHAVIOR environment check

Do not run BEHAVIOR / Isaac Sim checks on weak local machines. Use a simulation machine with a supported NVIDIA RTX GPU.

Minimum environment check must verify:

- Isaac Sim headless/offscreen startup.
- GPU and driver visibility.
- BEHAVIOR/OmniGibson import.
- Environment reset.
- One or more simulation steps.
- RGB observation extraction.
- Video or keyframe save.
- No snow screen, black screen, or segmentation fault.

Record output under:

```text
experiments/env_check/
logs/daily/YYYY-MM-DD.md
docs/env_setup.md
```

## How to run baseline rollout

Baseline rollout should be the smallest complete episode possible:

- Use a trivial policy, rule-based policy, or champion websocket policy.
- Save observation trace, action trace, reward/progress trace, success/failure flag, and video/keyframes.
- Record seed, task, machine, GPU, driver, CUDA/ROCm, Isaac Sim version, Python environment, checkpoint, and log path.

Output should go under:

```text
experiments/baseline_rollout/
docs/baseline_rollout.md
logs/daily/YYYY-MM-DD.md
```

## How to run cheap rollout

Cheap rollout mode must be explicitly named and logged. Candidate modes include:

- Low precision inference.
- Quantized model or NVFP4-like surrogate.
- Lower image resolution.
- Fewer denoising/action sampling steps.
- Smaller policy.
- Cached or approximate scoring policy.

Never assume cheap rollout is reliable. Always log speed, memory, success, reward/progress, stage metrics, and failure mode.

## How to run high-precision rollout

High-precision rollout is the reference mode for ranking. It should use the strongest feasible policy precision or the closest available trusted mode.

Record:

- dtype / precision mode.
- model checkpoint.
- inference latency.
- rollout wall-clock time.
- GPU memory.
- task, seed, candidate ID, and initial state.

## How to compute ranking correlation

For matching tasks, seeds, initial states, and candidate branches:

- Run cheap rollout for N candidates.
- Run high-precision rollout for the same candidate identities.
- Compute Spearman correlation.
- Compute Kendall tau.
- Compute top-k overlap.
- Compute success agreement.
- Compute false positive rate: cheap ranks good, high precision fails.
- Compute false negative rate: cheap ranks bad, high precision succeeds.
- Break down metrics by stage and horizon.

Store tables under `results/tables/` and plots under `results/figures/`.

## How to record experiments

Every experiment must have an ID:

```text
exp_YYYYMMDD_001
```

Record at least:

- Goal.
- Machine.
- GPU.
- Driver.
- CUDA or ROCm.
- Isaac Sim version.
- Python environment.
- Commit hash and dirty status.
- Config.
- Seed.
- Precision mode.
- Task.
- Number of episodes.
- Log path.
- Checkpoint path.
- Video path.
- Runtime.
- GPU memory.
- Failure mode.

## How to diagnose failures

Common early failures:

- Checkpoint load killed with exit code 137: likely system RAM or GPU memory pressure. Do not debug as Python logic unless a traceback exists.
- JAX sees only CPU: check whether command ran in sandbox or host environment.
- Websocket server not reachable: check host networking, `ss -ltnp`, firewall, SSH tunnel, and policy server bind host.
- BEHAVIOR mapping points to missing checkpoints: inspect `task_checkpoint_mapping*.json`.
- Isaac Sim black/snow screen: check GPU has RT cores, driver, headless/offscreen settings, container runtime, and Isaac Sim version.

## Common errors and fixes

- `~/models/checkpoint_*` missing: create or use a local mapping file pointing to actual checkpoint paths.
- Local 8GB GPU cannot load champion checkpoint: use a larger server; local machine is for code and docs only.
- `transformer_engine` missing in Sol-RL NVFP4 modes: install `transformer-engine[pytorch]` in the same Python interpreter as `torchrun`.
- HPSv2 reward checkpoint missing: place required files under `reward_ckpts/`.

## Paper-story alignment checklist

Before running an experiment, ask whether it answers at least one:

- Does it measure cheap-vs-high-precision agreement?
- Does it test horizon-driven ranking collapse?
- Does it isolate contact-rich failure modes?
- Does it compare naive cheap rollout scaling to verified/calibrated scaling?
- Does it support a main figure, table, ablation, or failure analysis?

If the answer is no, deprioritize the experiment.
