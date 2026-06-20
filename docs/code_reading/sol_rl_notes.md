# Sol-RL Notes

Last updated: 2026-06-19

## Repository Path / Link

- Local path: `/home/plote/longhorizon/rltask/Sana`
- Sol-RL docs in repo: `docs/sol_rl.md`
- Upstream project family: Sana / Sol-RL from NVLabs.

## Paper Summary

The local Sol-RL implementation is for diffusion post-training. The core advertised idea is FP4/NVFP4 explore and BF16 train. It uses low-cost preview rollout to search more candidates, then uses higher precision/full rollout for selected candidates and RL-style updates.

For this project, Sol-RL is an algorithmic reference, not a directly compatible mobile manipulation implementation.

## Core Algorithm

From `docs/sol_rl.md` and `configs/sol_rl/sana.py`:

- `diffusionnft`: PEFT-only baseline, 24-in-24.
- `naive_scaling`: brute-force PEFT scaling, 24-in-96.
- `compile`: BF16 compiled brute-force scaling, 24-in-96.
- `naive_quant`: direct NVFP4 compiled rollout, 24-in-96.
- `sol_rl`: two-stage decoupled rollout, 24-in-96.

For Sana Sol-RL:

- Stage 1 preview: `preview_step = 6`, `preview_model = "compile_nvfp4"`.
- Stage 2 full rollout: `fullrollout_model = "compile"`, `rollout_sample_num_steps = 10`.
- Selection mode defaults to `best_worst`.

## Training Loop

Main entry for Sana:

- `train_scripts/sol_rl/train_sana.py`

Single-node launcher:

- `train_scripts/sol_rl/run_sana_single_node_8gpu.sh`

Config:

- `configs/sol_rl/sana.py`
- `configs/sol_rl/base.py`

High-level loop observed:

1. Load model, PEFT/LoRA adapters, reward model, datasets, and distributed state.
2. Roll out multiple samples per prompt.
3. Save reward traces.
4. Compute advantages from grouped rewards.
5. Train on selected rollout samples for inner epochs.
6. Periodically save checkpoints and log eval metrics.

## Rollout Collection

Key function:

- `_rollout_for_one_prompt` in `train_scripts/sol_rl/train_sana.py`.

Behavior:

- If `preview_step > 0`, run draft rollouts using preview model for cheap scoring.
- Select seeds with `select_indices_by_mode`.
- Rerun selected seeds using full rollout model.
- Compute rewards asynchronously.
- Keep selected final samples for training.

This maps conceptually to our desired cheap rollout -> high-precision verification pipeline.

## Reward / Progress Signal

Current Sol-RL reward models are image/text reward functions, not robotics task rewards:

- PickScore.
- CLIPScore.
- HPSv2.
- ImageReward.

Relevant file:

- `diffusion/post_training/rewards.py`

For BEHAVIOR, these must be replaced with task success/progress/stage/contact metrics.

## Handling of Successful Trajectories

The local Sol-RL code groups rollouts by prompt and uses reward values to select/train. It does not directly model robotics success trajectories.

For our project, successful BEHAVIOR trajectories should become verified high-value samples, but failed trajectories need stage progress and failure-mode labels rather than being discarded.

## Handling of Failed Trajectories

The diffusion implementation keeps reward-scored samples and computes advantages over reward distributions. It does not have physical failure categories.

For BEHAVIOR, failed rollouts should be retained with:

- failure mode;
- stage reached;
- contact metrics;
- object progress;
- action saturation;
- timeout/collision/physics instability flags.

## Policy Update Objective

The local code uses a diffusion post-training objective with advantages and KL-like regularization controls. It is not directly a robot action policy objective.

Reusable ideas:

- grouped advantages;
- per-prompt/per-task statistics;
- reward trace serialization;
- decoupled rollout and train models;
- selection from many candidates before update.

Needs adaptation:

- action-policy loss for VLA/OpenPI;
- long-horizon credit assignment;
- sparse task success plus dense progress;
- stage-aware verification budget.

## Baselines

Sol-RL config families provide useful baseline structure:

- No scaling / DiffusionNFT-like baseline.
- Naive scaling.
- BF16 compiled scaling.
- Naive quantized rollout.
- Two-stage Sol-RL rollout.

For this project, equivalent baselines should be:

- No RL / champion baseline.
- High-precision rollout only.
- Naive cheap rollout scaling.
- Cheap rollout plus high-precision verification.
- Stage-aware calibrated rollout scaling.

## Evaluation Protocol

Sol-RL evaluates generation quality with reward models and logs reward means/ranges.

For BEHAVIOR, evaluation must use:

- success rate;
- reward/progress;
- stage completion;
- contact failure metrics;
- runtime and resource cost;
- ranking correlation against high-precision rollout.

## Code Structure

- `docs/sol_rl.md`: usage guide.
- `train_scripts/sol_rl/train_sana.py`: Sana training loop.
- `train_scripts/sol_rl/train_sd3.py`: SD3 training loop.
- `train_scripts/sol_rl/train_flux1.py`: FLUX.1 training loop.
- `train_scripts/sol_rl/train_utils.py`: distributed helpers, reward trace serialization, logging, selection helpers.
- `configs/sol_rl/base.py`: shared config.
- `configs/sol_rl/sana.py`: Sana experiment families.
- `configs/sol_rl/sd3.py`: SD3 experiment families.
- `configs/sol_rl/flux1.py`: FLUX.1 experiment families.
- `diffusion/post_training/rewards.py`: reward model dispatch.

## Main Entry Points

```bash
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

Config selection:

```bash
CONFIG_SPEC=configs/sol_rl/sana.py:sana_sol_rl_pickscore \
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
```

## Config System

- Python config modules expose `get_config(name)`.
- `ml_collections.ConfigDict` is used.
- Named functions define experiment families.
- Important fields: `preview_step`, `preview_model`, `fullrollout_model`, `sample.num_image_per_prompt`, `sample.best_of_n`, `sample.full_rollout_num`, `sample.rollout_batch_size`, and reward function names.

## How to Run Minimal Demo

Not run in this workspace.

The documented minimal Sol-RL run requires multi-GPU training and external model/reward dependencies. It should not be run locally without approval.

Safer first validation:

- Import config.
- Print one config object.
- Inspect reward model dependency availability.
- Do not start `torchrun` until resources are confirmed.

## Components We Can Reuse

- Two-stage candidate selection design.
- Naming convention for baselines: naive scaling, naive quant, verified/decoupled rollout.
- Reward trace JSON structure concept.
- Per-prompt grouped advantage logic, adapted to per-task/per-initial-state groups.
- Time logging and distributed training structure as reference.

## Components We Need to Modify

- Replace image reward models with BEHAVIOR progress/success/contact metrics.
- Replace diffusion prompt/sample abstraction with BEHAVIOR task/seed/candidate abstraction.
- Replace image generation rollout with environment rollout.
- Replace diffusion policy update objective with VLA/OpenPI-compatible update.
- Add stage-aware verification and ranking correlation metrics.

## How It Can Connect to BEHAVIOR

Conceptual mapping:

- Prompt -> task_id + task instance + seed.
- Image sample seed -> rollout candidate ID / branch seed.
- Preview rollout -> cheap BEHAVIOR rollout mode.
- Full rollout -> high-precision BEHAVIOR rollout mode.
- Image reward -> task progress/success/stage/contact score.
- Reward trace -> rollout record JSON.
- Advantage -> per-task candidate advantage for policy update.

## How It Can Support Cheap Rollout Scaling

Sol-RL provides the structure for:

- generating many cheap candidates;
- selecting candidates for high-precision rerun;
- storing reward traces;
- computing candidate advantages;
- comparing naive cheap scaling to verified scaling.

Our project must add reliability analysis before trusting cheap candidates.

## Open Questions

- Whether Sol-RL's update objective can be reused or only its rollout-selection structure.
- Which robotics RL objective should be paired with OpenPI/Pi0.5: PPO, GRPO-like, DPO/preference, behavior cloning from verified successes, or advantage-weighted regression.
- Whether cheap rollout should be lower precision inference or a different surrogate.
- How to preserve candidate identity between cheap and high-precision BEHAVIOR rollouts.
