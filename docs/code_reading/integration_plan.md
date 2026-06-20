# Integration Plan: OVMM x BEHAVIOR Champion Solution x Sol-RL x Our LMM Project

Last updated: 2026-06-20

## Goal

Build a minimal pipeline where OVMM provides the first runnable mobile-manipulation environment loop, BEHAVIOR champion provides the later long-horizon benchmark/policy baseline, Sol-RL provides the online RL / rollout scaling design reference, and this project adds cheap-vs-high-precision ranking reliability analysis with stage-aware verification.

Final target:

Low-cost rollout scaling can improve online RL for long-horizon mobile manipulation only when its ranking reliability is explicitly modeled and corrected, especially under long horizons and contact-rich stages.

## Scope Reset

This is the active plan, not a continuation of the older `Planning/Doc/Todo.md` route.

- Old OpenVLA experiments are historical reference only.
- Old J-EPA/V-JEPA reward ideas are not part of the current mainline.
- The immediate smoke target is OVMM/HomeRobot.
- The later long-horizon baseline is the BEHAVIOR champion solution.
- The current Sol-RL dependency is conceptual: cheap preview / high-fidelity verification / rollout trace / update loop structure.

## Current Role of Each Codebase

### BEHAVIOR Champion Solution

- Path: `/home/plote/longhorizon/basecode/behavior-1k-solution`
- Role: environment/policy baseline, websocket policy server, checkpoint loading, task-conditioned execution, stage wrapper.
- Current runnable status: not runnable on local workstation due to OOM during checkpoint restore.

### OVMM / HomeRobot

- Path: `/home/plote/longhorizon/benchmark/home-robot`
- Role: first environment loop for local mobile-manipulation smoke tests and rollout-schema prototyping.
- Current runnable status: Python stack and OVMM episodes are available locally; simulator initialization is blocked by local OpenGL/EGL context failure.

### Sol-RL / Sana

- Path: `/home/plote/longhorizon/rltask/Sana`
- Role: algorithm reference for cheap preview rollout, high-precision/full rollout, selection, reward traces, and online update loop.
- Current runnable status: not tested locally; likely requires multi-GPU setup and reward/model assets.

### Our LMM Project Workspace

- Path: `/home/plote/longhorizon`
- Role: umbrella workspace for memory, docs, experiment logs, integration scripts, results, and future glue code.
- Current runnable status: documentation initialized; no integrated rollout code yet.

## Stage 0: Read and Document

Status: in progress.

- Read prior notes in `Planning/Doc/Todo.md`.
- Do not inherit the prior OpenVLA/J-EPA plan as the current plan.
- Read BEHAVIOR champion README and key policy server/wrapper files.
- Read Sol-RL docs, configs, rollout function, and shared utilities.
- Create project memory, skill, paper story, code notes, and integration plan.
- Identify dependency and resource risks.

## Stage 1A: Run Minimal OVMM Demo

Goal:

- Start HomeRobot/Habitat OVMM.
- Reset one task.
- Step one random or heuristic episode.
- Save logs and minimal rollout metadata.

Current local status:

- `home-robot` env exists.
- `habitat_sim==0.2.5`, `habitat`, `home_robot`, and `home_robot_sim` import.
- OVMM episodes exist at `data/datasets/ovmm`.
- Local simulator creation fails with `GL::Context: cannot retrieve OpenGL version`.

Required outputs:

- `docs/env_setup.md`
- `experiments/env_check/`
- `logs/daily/YYYY-MM-DD.md`

## Stage 1B: Run Minimal BEHAVIOR Demo

Goal:

- Start BEHAVIOR/OmniGibson on a simulation-capable machine.
- Reset one task.
- Step one episode with a trivial or existing policy.
- Save logs and video/keyframes.

Required outputs:

- `docs/env_setup.md`
- `docs/baseline_rollout.md`
- `experiments/env_check/`
- `experiments/baseline_rollout/`
- `logs/daily/YYYY-MM-DD.md`

Do not start with all 50 tasks.

## Stage 2: Wrap OVMM and BEHAVIOR into a Standard Rollout Interface

Define a unified rollout API:

```python
rollout(
    policy,
    env,
    task_id,
    seed,
    precision_mode,
    max_steps,
    log_dir,
)
```

The rollout output must include:

```python
{
    "task_id": str,
    "seed": int,
    "precision_mode": str,
    "success": bool,
    "reward": float,
    "progress": float,
    "stage_metrics": dict,
    "failure_mode": str,
    "actions": path,
    "observations": path,
    "video": path,
    "runtime": float,
    "gpu_memory": float,
}
```

Additional fields recommended:

```python
{
    "experiment_id": str,
    "candidate_id": str,
    "initial_state_id": str,
    "checkpoint_path": str,
    "policy_server_host": str,
    "simulator_version": str,
    "commit_hashes": dict,
    "dirty_repos": dict,
}
```

## Stage 3: Connect Sol-RL Training Loop

Initial target:

- Do not train yet.
- Convert BEHAVIOR rollout records into Sol-RL-like reward traces.
- Reproduce grouped candidate scoring and advantage calculation offline.

Later target:

- Start with high-precision rollout only.
- Confirm training loop mechanics before adding cheap rollout.
- Add cheap rollout only after baseline logging and replay are stable.

## Stage 4: Add Cheap Rollout Mode

Candidate cheap modes to evaluate:

- Lower precision policy inference.
- Quantized policy or converted PyTorch policy.
- Fewer denoising/action sampling steps.
- Lower image resolution.
- Smaller surrogate policy.
- Stage-specific cheap heuristic or scoring model.

Every cheap mode must log:

- latency;
- GPU memory;
- rollout wall-clock;
- success/progress;
- stage metrics;
- failure mode;
- compatibility with candidate replay.

## Stage 5: Run Ranking Reliability Experiment

For matching task seeds and candidate identities:

- Run cheap rollout.
- Run high-precision rollout.
- Compute Spearman correlation.
- Compute Kendall tau.
- Compute top-k overlap.
- Compute success agreement.
- Compute false positive and false negative cases.
- Break down by navigation, approach, grasp/contact, articulation, placement, and completion.

Expected outputs:

- `docs/ranking_experiment.md`
- `experiments/ranking_corr/`
- `results/tables/ranking_corr.csv`
- `results/figures/ranking_corr_*.png`

## Stage 6: Add Stage-Aware Verification

Method outline:

- Cheap rollout proposes candidates.
- Stage risk or uncertainty decides which candidates get high-precision verification.
- Verified rollouts enter RL update.
- Compare against naive cheap rollout scaling under matched budgets.

Risk features:

- contact duration;
- contact lost count;
- object joint progress;
- action saturation;
- gripper state changes;
- horizon length;
- stage transition confidence;
- cheap score uncertainty.

## Minimal Closed Loop

1. On model server, load champion checkpoint and start websocket policy.
2. On simulation server, run one BEHAVIOR evaluator episode against that policy.
3. Save standardized rollout record.
4. Parse the record into a table.
5. Repeat with a second precision/cheap mode for the same task/seed/candidate where feasible.
6. Compute one small ranking agreement diagnostic.

## First Week Development Plan

Day 1:

- Finish memory/docs/notes.
- Confirm machine roles and available remote paths.
- Make environment checklist for BEHAVIOR sim and policy server.

Day 2:

- Run Isaac Sim / BEHAVIOR headless smoke test on simulation machine.
- Save keyframe or short video.
- Record exact environment versions.

Day 3:

- Load `checkpoint_1` on large model server.
- Start websocket policy server.
- Confirm evaluator can connect.

Day 4:

- Run one minimal task instance.
- Save full baseline rollout metadata.
- Start failure taxonomy.

Day 5:

- Add or write a rollout record parser.
- Define stage metrics available from logs.
- Produce first baseline rollout report.

Day 6:

- Identify first cheap rollout mode.
- Run microbenchmarks for latency/memory if safe.

Day 7:

- Design first ranking correlation experiment for a tiny task subset.
- Review whether evidence still supports the BEHAVIOR mainline.

## Current Largest Risks

- BEHAVIOR/Isaac Sim may be unstable on available simulation hardware.
- Champion checkpoints require more memory than local machine; remote server must be validated.
- Cheap rollout mode for JAX/OpenPI may be nontrivial.
- Sol-RL code is diffusion-specific; only its rollout-selection structure may transfer directly.
- Without reliable stage/contact metrics, the paper claim about contact-rich collapse will be weak.

## Dependency Conflict Notes

- BEHAVIOR champion uses JAX/OpenPI/Orbax and OmniGibson/Isaac Sim evaluator.
- Sol-RL/Sana uses PyTorch, diffusers-style pipelines, Transformer Engine for NVFP4, and image reward models.
- Treat them as separate environments at first. Do not merge dependencies until the minimal data interface is stable.

## Glue Code Needed

- `rollout_record` schema and serializer.
- BEHAVIOR evaluator log parser.
- Stage/contact metric extractor.
- Candidate identity and seed management.
- Cheap/high-precision run launcher.
- Ranking correlation analysis script.
- Sol-RL-like reward trace adapter for BEHAVIOR rollout records.
- Optional policy update adapter after baseline and ranking experiments are stable.
