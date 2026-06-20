# Paper Story: LMM Rollout Scaling

Last updated: 2026-06-19

## Working Title

LMM: Online Reinforcement Learning with Rollout Scaling for Long-Horizon Mobile Manipulation

## One-Sentence Thesis

Low-cost rollout scaling can improve online RL for long-horizon mobile manipulation only when cheap rollout ranking reliability is explicitly measured and corrected, especially under long horizons and contact-rich stages.

## Motivation

Long-horizon mobile manipulation has high rollout cost, sparse success signals, and late-stage failures. Online RL can in principle improve policies, but collecting enough high-fidelity rollouts in BEHAVIOR-like environments is expensive. Cheap rollout promises more exploration, but it may become misleading when precision errors accumulate or contact dynamics dominate.

## Core Research Question

When does cheap rollout preserve high-precision rollout ranking, when does it collapse, and can stage-aware verification or calibration make cheap rollout useful for online RL?

## Hypotheses

- H1: Cheap rollout is not uniformly reliable.
- H2: Contact-rich stages are the main reliability bottleneck.
- H3: Naive cheap rollout scaling can harm or fail to improve online RL when ranking collapses.
- H4: Stage-aware verification or calibration can recover rollout scaling utility.

## Anchor Work Alignment

### BEHAVIOR Champion Solution

Role: environment and engineering baseline.

It provides:

- BEHAVIOR-1K long-horizon mobile manipulation evaluation.
- A Pi0.5/OpenPI-derived policy server.
- 4 task-specialized checkpoints.
- Stage tracking, correction rules, action compression, and rolling inpainting.
- A realistic starting point for failure mode analysis.

### Sol-RL

Role: online RL and rollout-scaling algorithm reference.

It provides:

- A two-stage rollout design: cheap preview rollout followed by higher precision full rollout.
- Reward trace logging.
- Advantage computation and policy update loop.
- Baselines for naive scaling, quantized rollout, and decoupled rollout.

Our difference:

- Sol-RL studies diffusion post-training with FP4 rollout and BF16 training.
- This project studies long-horizon mobile manipulation, where cheap rollout reliability must be analyzed by stage, horizon, and contact dynamics.

## Current Scope Decision

The active route is champion-solution-first:

- Use the BEHAVIOR champion solution as the main policy and evaluation baseline.
- Treat older OpenVLA-centered notes as historical exploration, not the current baseline.
- Do not include J-EPA/V-JEPA/world-model reward in the main method unless the project explicitly pivots later.
- Use Sol-RL as a rollout-scaling pattern, not as a requirement to adopt its original diffusion-specific model stack.

## Proposed Main Claims

### Claim 1: Cheap rollout ranking reliability degrades with horizon.

Evidence needed:

- Spearman/Kendall/top-k overlap curves by horizon length.
- Same initial states and candidate identities evaluated in cheap and high-precision modes.

### Claim 2: Contact-rich stages cause disproportionate ranking collapse.

Evidence needed:

- Stage-level correlation metrics.
- Contact lost count, action saturation, object joint progress, and failure case videos.
- False positive examples where cheap rollout ranks a candidate highly but high precision fails.

### Claim 3: Naive cheap rollout scaling is not enough.

Evidence needed:

- Compare high-precision-only, naive cheap rollout, and cheap rollout plus verification.
- Show cases where more cheap samples do not translate to better high-precision success or RL improvement.

### Claim 4: Stage-aware verification/calibration recovers useful rollout scaling.

Evidence needed:

- A verification policy that spends high-precision rollout budget on risky or high-potential candidates.
- Improved success, sample efficiency, and wall-clock/GPU-hour efficiency.

## Main Experimental Progression

1. Establish environment and baseline reproducibility.
2. Run champion baseline and collect stage/failure logs.
3. Define cheap and high-precision modes.
4. Measure ranking correlation by task, stage, horizon, and contact type.
5. Design stage-aware verification or calibration only after observing failure patterns.
6. Integrate verified rollout into online RL.
7. Compare against no RL, high-precision-only, naive cheap rollout, and verified cheap rollout.

## Figure / Table Requirements

- Figure: system overview connecting BEHAVIOR rollout, cheap candidate generation, high-precision verification, and online RL update.
- Figure: ranking correlation vs horizon.
- Figure: ranking correlation by stage.
- Figure: failure case panels for contact-rich collapse.
- Table: environment and resource cost per rollout mode.
- Table: online RL comparison across baseline, high-precision-only, naive cheap, and verified cheap.
- Table: false positive/false negative rates by stage.

## Current Evidence

- Engineering evidence only: local workstation cannot load champion checkpoint due to OOM.
- No rollout correlation or RL results yet.

## Current Non-Claims

- We cannot yet claim cheap rollout works.
- We cannot yet claim contact stages collapse.
- We cannot yet claim BEHAVIOR baseline is reproduced.
- We cannot yet claim online RL improvement.
- We are not claiming an OpenVLA result.
- We are not claiming a J-EPA/V-JEPA reward or world-model contribution.

## Immediate Paper Risk

- If BEHAVIOR/Isaac Sim is unstable, main benchmark execution may be blocked.
- If no feasible cheap rollout mode exists for the champion policy, the project must use a surrogate cheap mode or pivot benchmark/model.
- If cheap and high-precision rankings are always strongly correlated, the novelty shifts toward efficiency rather than reliability correction.
- If rankings are uncorrelated everywhere, cheap rollout may be unusable and the method needs stronger calibration or verification.

## Next Evidence Needed

1. Minimal BEHAVIOR environment check on a simulation machine.
2. Champion policy server load on a large model server.
3. One complete baseline episode with logs and video.
4. A stage/failure annotation schema.
5. A concrete cheap rollout mode proposal for the BEHAVIOR policy.
