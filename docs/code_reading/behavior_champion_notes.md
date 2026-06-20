# BEHAVIOR Champion Solution Notes

Last updated: 2026-06-19

## Repository Path / Link

- Local path: `/home/plote/longhorizon/basecode/behavior-1k-solution`
- Upstream described in README: `https://github.com/IliaLarchenko/behavior-1k-solution`

## Environment Requirements

- Python: project requires Python >=3.11,<3.12.
- Package manager: README uses `uv`; local `.venv` exists.
- Major dependencies: OpenPI, JAX CUDA, Flax, Orbax, Torch, OpenCV, LeRobot, WandB, HuggingFace Hub.
- Simulation/eval dependency: BEHAVIOR-1K / OmniGibson / Isaac Sim environment.
- Practical hardware requirement: local RTX 3070 Ti Laptop GPU with 8GB VRAM is not enough to load champion checkpoints. Use larger policy server hardware.
- Simulation hardware risk: Isaac Sim rendering is sensitive to GPU/driver/headless configuration. Prior notes emphasize RTX/RT-core GPUs for simulation and warn about snow screen on some server GPUs.

## Main Entry Points

- Policy server: `scripts/serve_b1k.py`
- Training: `scripts/train.py`
- Norm stats: `scripts/compute_norm_stats.py`
- FAST tokenizer training: `scripts/train_fast_tokenizer.py`
- Evaluation client is external BEHAVIOR/OmniGibson:

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  task.name=make_microwave_popcorn \
  model.host=localhost \
  eval_instance_ids="[0,1,2,3]"
```

## Code Structure

- `src/b1k/training/config.py`: training/data/model config for `pi_behavior_b1k_fast`.
- `src/b1k/policies/policy_config.py`: trained policy loading, checkpoint restore, transform pipeline.
- `src/b1k/policies/b1k_policy.py`: BEHAVIOR observation/action transforms.
- `src/b1k/shared/eval_b1k_wrapper.py`: policy wrapper used by BEHAVIOR evaluator observations.
- `src/b1k/policies/checkpoint_switcher.py`: maps task IDs to one of 4 checkpoints and lazily switches policy.
- `src/b1k/shared/correction_rules.py`: hardcoded eval correction rules.
- `openpi/`: bundled OpenPI dependency.
- `BEHAVIOR-1K-684a/`: local BEHAVIOR/OmniGibson-related checkout or extracted tree.
- `checkpoints/`: 4 downloaded champion checkpoints.

## Task / Environment Interface

- The policy server exposes a websocket policy interface compatible with BEHAVIOR evaluator.
- The evaluator sends observation dicts with BEHAVIOR/OmniGibson keys.
- The wrapper extracts `task_id` from observations when present and uses it for task-conditioned inference and checkpoint switching.
- The policy itself is task-conditioned by task ID and current stage, not natural-language prompts.

## Observation Space

Observed in `B1KPolicyWrapper.process_obs`:

- `robot_r1::proprio`
- `robot_r1::robot_r1:zed_link:Camera:0::rgb`
- `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb`
- `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb`

The wrapper resizes RGB images to 224x224 and maps them to:

- `observation/egocentric_camera`
- `observation/wrist_image_left`
- `observation/wrist_image_right`
- `observation/state`

## Action Space

- The model action dimension is configured as 32.
- Output transform returns the first 23 action dimensions for BEHAVIOR execution.
- The wrapper outputs a Torch float tensor.
- Policy execution uses action chunks: predict 30, execute 26, keep 4 for inpainting, and optionally compress 26 actions into 20 steps.

## Reward / Success Definition

- Not implemented in this repository's policy server.
- Reward and success are owned by BEHAVIOR/OmniGibson evaluator.
- For our project, evaluator logs must be inspected to extract success, progress, stage transitions, and failure modes.

## Execution Pipeline

1. Start policy server with `scripts/serve_b1k.py`.
2. Load `pi_behavior_b1k_fast` config.
3. Restore checkpoint via `policy_config.create_trained_policy`.
4. Create optional `CheckpointSwitcher` from mapping JSON.
5. Wrap policy in `B1KPolicyWrapper`.
6. BEHAVIOR evaluator connects by websocket.
7. For each observation, wrapper processes RGB/proprio, inserts task/stage state, calls policy inference, applies correction/compression/inpainting, and returns an action.

## Stage Decomposition

The champion model has an internal stage prediction and wrapper-side stage voting. Exact semantic labels per task still need to be mapped from BEHAVIOR metadata and task definitions.

- Navigation: likely implicit in task stages; needs evaluator/task metadata inspection.
- Object search: likely implicit; needs task metadata.
- Approach: likely implicit; needs task metadata.
- Grasp: contact-rich; correction rules and gripper variation checks are relevant.
- Articulation: contact-rich; object joint progress should be added to our logs.
- Placement: contact-rich; release/contact-lost failures should be tracked.
- Completion: BEHAVIOR evaluator success condition.

## Logging / Evaluation

- Policy server logs task changes, stage transitions, correction rules, and periodic step/prediction status.
- Evaluation logs are external to the policy server and should be stored in experiment directories.
- Our project needs a wrapper/log parser that records task, seed, checkpoint, precision mode, stage metrics, action trace, observation trace, video/keyframes, runtime, and failure mode.

## Existing Models / Checkpoints

Downloaded locally:

- `checkpoints/checkpoint_1`: tasks 2, 3, 5, 6, 10, 11, 13, 14, 15, 19, 23, 24, 25, 28, 29, 34, 42, 44, 47, 48.
- `checkpoints/checkpoint_2`: tasks 0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43, 45.
- `checkpoints/checkpoint_3`: tasks 4, 27, 31, 32, 33, 35, 36, 37, 38, 39, 41, 46, 49.
- `checkpoints/checkpoint_4`: task 40.

Local size: about 12GB each, 48GB total.

## How to Run Minimal Demo

Requires a larger policy server than the local workstation:

```bash
cd /path/to/behavior-1k-solution
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ./.venv/bin/python scripts/serve_b1k.py \
  --task-checkpoint-mapping task_checkpoint_mapping.local.json \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_fast \
  --policy.dir checkpoints/checkpoint_1
```

Then run BEHAVIOR eval client on the simulation machine:

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  task.name=make_microwave_popcorn \
  model.host=<policy_server_host> \
  eval_instance_ids="[0]"
```

## Known Errors

- Local machine cannot load `checkpoint_1`; process killed with signal 9 / exit code 137 at about 9.86GB RSS.
- Original mapping file points to nonexistent `~/models/checkpoint_*` locally.
- JAX GPU visibility depends on host execution; sandboxed commands saw CPU only.
- BEHAVIOR/Isaac Sim rendering may fail with snow/black screen depending on GPU and driver.

## Components We Can Reuse

- BEHAVIOR evaluator/websocket interface.
- Champion policy server.
- Observation transform and action transform.
- Checkpoint switcher.
- Stage tracking and stage-voting logic.
- Correction rule hooks as baseline behavior.
- Existing checkpoints as baseline policies.

## Components We Need to Replace

- Logging and experiment metadata are insufficient for this project's ranking reliability experiments.
- Stage semantics need to be exposed as research metrics, not only internal policy state.
- Cheap/high-precision rollout modes need to be added or wrapped.
- Online RL update loop is not present in this repo.

## Where to Insert VLA Policy

- Existing insertion point: `policy_config.create_trained_policy` and `B1KPolicyWrapper.policy`.
- A new VLA policy should implement an `infer` or `act` interface compatible with `B1KPolicyWrapper`.

## Where to Insert Cheap Rollout Mode

Candidate insertion points:

- Alternative policy loader in `policy_config.py`.
- Wrapper-level precision mode switch in `serve_b1k.py`.
- Separate policy server for cheap mode, with the evaluator selecting host/port.
- Offline rollout script that records candidate actions and replays/evaluates them.

## Where to Insert High-Precision Rollout Mode

- The existing champion checkpoint path can be treated as high-precision baseline if it loads with BF16/JAX on a large server.
- A high-precision server should emit the same rollout record schema as cheap mode.

## Where to Insert Online RL Update

- Not in the current policy server.
- Build a separate training loop that consumes standardized rollout records and produces updated policy checkpoints.
- Sol-RL should be treated as algorithm reference for rollout collection, reward tracing, advantage estimation, and update scheduling.

## Open Questions

- Exact BEHAVIOR version/API expected by `BEHAVIOR-1K-684a`.
- How to extract fine-grained stage labels and contact metrics from evaluator state.
- Which tasks should be the first minimal benchmark subset.
- Which cheap rollout mode is feasible for JAX/OpenPI champion policy.
- Whether checkpoint switching is reliable across tasks under memory pressure on a large server.
