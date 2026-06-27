# Current Plan: OVMM-First LMM Rollout Scaling

Last updated: 2026-06-27

## Purpose

This file is the active project plan. It supersedes the older exploration notes in `Planning/Doc/Todo.md` for day-to-day decisions.

The old Todo is useful historical context, but it came from a different exploration direction: OpenVLA-centered policy work and J-EPA/V-JEPA-style reward or world-representation ideas. That is not the current route.

On 2026-06-20 the immediate engineering route shifted to OVMM first because it is simpler to bring up than BEHAVIOR/Isaac. BEHAVIOR champion remains the stronger long-horizon platform for the final paper story, but the first local smoke loop should be OVMM.

## Active Route

Use OVMM/HomeRobot to get the first mobile-manipulation rollout loop running, then transfer the rollout logging and ranking-reliability interface to the BEHAVIOR champion solution.

Current active stack:

- First smoke benchmark/environment: OVMM/HomeRobot in Habitat.
- Later main long-horizon benchmark/environment: BEHAVIOR-1K with the champion solution.
- Baseline policy for OVMM: heuristic/random/skill-based HomeRobot agents first.
- Baseline policy for BEHAVIOR: BEHAVIOR champion solution checkpoints and websocket policy server.
- Algorithm reference: Sol-RL for the rollout-scaling pattern, not as a drop-in robotics training loop.
- Core experiment: cheap rollout ranking reliability vs high-precision rollout, broken down by horizon and contact-rich stage.
- Method direction: stage-aware verification or calibration if ranking collapse is observed.

## Explicit Non-Goals

- Do not treat OpenVLA as the primary baseline right now.
- Do not add J-EPA/V-JEPA/world-model reward to the mainline.
- Do not start by designing SRPO or latent representation rewards.
- Do not run large training before the champion baseline and BEHAVIOR evaluation pipeline are reproducible.
- Do not merge Sana/Sol-RL dependencies into the champion repo until the data interface is defined.

## Why OVMM First

OVMM is the first practical target because:

- It already has a local `home-robot` conda environment.
- It uses Habitat rather than Isaac Sim, so the setup surface should be smaller.
- It can provide an earlier reset/step/log loop for mobile manipulation.
- It is enough to prototype rollout records, cheap-vs-high-precision scoring tables, and Sol-RL-style selection logic.

## Why Champion Still Matters

The BEHAVIOR champion solution still provides:

- A BEHAVIOR-specific Pi0.5/OpenPI-derived policy.
- Four task-specialized checkpoints.
- Task ID conditioning and stage tracking.
- BEHAVIOR-compatible websocket serving.
- Existing inference engineering: rolling inpainting, action compression, correction rules, and checkpoint switching.

This makes it the better later platform for BEHAVIOR long-horizon mobile manipulation than rebuilding from OpenVLA.

## What Sol-RL Contributes

Sol-RL contributes the abstract rollout-scaling pattern:

1. Generate many cheap candidates.
2. Select candidates using a cheaper score or preview.
3. Rerun or verify selected candidates with a higher-fidelity mode.
4. Use verified data for policy improvement.

For this project, Sol-RL is not copied directly. It must be translated from image diffusion post-training to BEHAVIOR rollout records and robot policy updates.

Important clarification from 2026-06-27:

- Sol-RL's FP4/NVFP4 implementation is not a required method for this project.
- Transformer Engine / NVFP4 is an image-generation-specific cheap rollout mechanism, not the scientific contribution.
- Our project only needs a cheap rollout proxy that can be compared against high-precision rollout.
- For VLM / flow-matching / action-chunk policies, cheap rollout can come from quantized weights, fewer action sampling or flow integration steps, lower observation resolution, cached visual/language features, smaller draft policies, or other lower-cost rollout approximations.
- Robotics reward/progress should come from environment success, stage progress, contact/articulation metrics, and verified rollout outcomes, not Sana image reward models.

## First Minimal Closed Loop

0. Use local HomeRobot/OVMM as the first environment target.
1. Fix or bypass local Habitat-Sim EGL/OpenGL initialization.
2. Run one OVMM random or heuristic episode.
3. Save action, observation, success/progress, runtime, and failure mode.
4. Wrap OVMM rollout output into the same standard rollout schema planned for BEHAVIOR.
5. Move to BEHAVIOR champion once NVIDIA rendering and checkpoint serving are available.

## First Research Milestone

After baseline rollout works:

1. Define a cheap rollout mode for the champion policy.
2. Run matched candidate rollouts under cheap and high-precision modes.
3. Compute ranking correlation and false positive/false negative rates.
4. Break down results by stage and horizon.

## Current Risks

- Local OVMM currently reaches dataset initialization but fails at Habitat-Sim OpenGL/EGL context creation.
- Local machine cannot load the BEHAVIOR champion checkpoint.
- BEHAVIOR/Isaac Sim may be unstable on available simulation hardware.
- AMD training compatibility is not guaranteed because current champion/Sana dependency files are CUDA-oriented.
- Cheap rollout mode for the champion policy is still undefined.
- Stage/contact metrics must be exposed from BEHAVIOR logs before the paper claim is testable.

## Decision Log

- 2026-06-19: Active plan reset to champion-solution-first. Prior OpenVLA/J-EPA exploration is historical reference only.
- 2026-06-20: Immediate Stage 1 target changed to OVMM-first per advisor guidance. BEHAVIOR champion remains the later/main long-horizon target.
- 2026-06-27: NVFP4/Transformer Engine downgraded to optional engineering detail. Active cheap rollout design should be robotics-policy-specific rather than copied from image-generation Sol-RL.
