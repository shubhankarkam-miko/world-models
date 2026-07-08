# SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model

## TL;DR

SafeDojo is the first model-based safe reinforcement learning framework for Vision-Language-Action (VLA) policies. Instead of exploring unsafe actions in the real world or hand-coding safety rules, it trains an action-conditioned video world model to *imagine* candidate rollouts, scores each imagined rollout with two separate learned heads (a task-success classifier and a collision-risk classifier), and optimizes the policy with a Lagrangian-constrained variant of GRPO that treats task reward and safety cost as two independent signals rather than one hand-weighted scalar. On SafeLIBERO (a collision-augmented version of LIBERO) it beats CBF-based inference-time filtering, model-free constrained RL, and other world-model-based VLA post-training methods on the safe-success rate metric, and the gains transfer to five real-world Franka tasks.

## Architecture

The system has three moving parts, all built around a single core idea: **do RL rollouts inside an imagined video, not the real world, and evaluate those imagined rollouts with two independent heads rather than one combined reward.**

1. **Interactive world model (Section 4.1).** An action-conditioned video predictor $p_\phi$, built on top of Wan 2.2 (a large open video diffusion/DiT model) with cross-attention action conditioning. Given the current imagined observation $\hat o_{i,k}$, latent $\hat z_{i,k}$, an action chunk $a_{i,k}$, and the language instruction $l$, it predicts the next chunk's observation and latent: $(\hat o_{i,k+1}, \hat z_{i,k+1}) = p_\phi(\hat o_{i,k}, \hat z_{i,k}, a_{i,k}, l)$ (Eq. 2). Action chunks are $H=8$ control steps of a 7-DoF command (6-DoF delta end-effector pose + gripper), matching the OpenVLA-OFT action interface. Only the DiT component of Wan is fine-tuned; the VAE is frozen. A "static-video augmentation" (occasionally freezing conditioning frames during training) is added specifically to combat *exposure error* — the compounding mismatch between clean training observations and the model's own imperfect self-generated observations during autoregressive rollout.

2. **Dual-branch reward-cost evaluator (Section 4.2).** Rather than collapsing task progress and safety into one hand-crafted scalar reward, SafeDojo estimates them with two separate, independently trained heads applied to every imagined trajectory:
   - $f_\text{task}$: a ResNet-style binary success classifier operating on RGB frames, fine-tuned on rollout frames with per-frame success labels, giving a dense per-step task-progress reward $r_{i,h}$.
   - $f_\text{safe}$: a compact convolutional latent encoder + action-conditioned MLP head operating on Wan's *latent* features (not RGB), outputting a per-step obstacle-contact probability $c_{i,h}$ for the proposed action chunk.
   
   These per-step scores are aggregated into trajectory-level scalars (Eq. 4): task reward is a plain average over all $KH$ steps, but safety cost is a **top-M average** (M=16) over the highest-risk steps only — deliberately emphasizing brief dangerous intervals rather than diluting them across a long, mostly-safe trajectory.

3. **Lagrangian-constrained GRPO (Section 4.3).** Building on GRPO (Section 3.2: group-relative advantage normalization without a learned value critic), SafeDojo normalizes reward and cost *separately* within a sampled group of $G$ trajectories (Eq. 5), then combines them via a Lagrange multiplier $\eta$: $\hat A^\text{safe}_i = \hat A^r_i - \eta \hat A^c_i$ (Eq. 6). Critically, $\eta$ is not fixed — it's updated every batch based on whether the *average* safety cost across the group exceeds a fixed budget $d=0.2$ (Eq. 7): $\eta$ rises when the policy is too unsafe on average, and falls otherwise, letting the constraint pressure self-adjust rather than requiring a hand-tuned penalty weight $\lambda$ (which the paper explicitly contrasts against as a naive baseline: $\tilde r_i = r_i - \lambda c_i$).

The full loop (Algorithm 1): sample a group of $G=16$ candidate trajectories from the current policy → roll each through the world model → score with the two heads → aggregate → normalize advantages → update $\eta$ → compute the constrained advantage → update the policy by the clipped GRPO objective (Eq. 8, with a KL term to a reference policy that is set to zero weight in practice, relying instead on GRPO's clip ratio for trust-region control).

The base VLA policy itself is not novel — it's OpenVLA-OFT (7B, token/continuous hybrid action head), used unmodified as the policy being trained; SafeDojo's contribution is entirely in how that policy is optimized, not in the policy architecture itself.

## Data

- **World-model fine-tuning data:** 1.5K SafeLIBERO trajectories, converted to a "Wan/RLinf video-action format" — 13 video frames at 256×256 per training sample, 5 conditioning frames plus action conditioning as extra input.
- **Task reward head data:** rollout frames labeled with per-frame binary success signals (positive-class weighting used because successful frames are sparse).
- **Safety cost head data:** SafeLIBERO's built-in "contact-collision auditing" — for each rollout chunk, context latents, the candidate action chunk, binary per-step contact labels, and a valid-step mask; split at the episode level (not step level) to prevent chunk leakage between train/val.
- **Evaluation benchmark:** SafeLIBERO [18] — a safety-augmented version of LIBERO [42] built in Robosuite [43], with four suites (Spatial, Goal, Object, Long) × two obstacle-interference levels (Level I: obstacles near the target object; Level II: obstacles farther away but along the motion path). All methods are trained on Level I only; Level II is a zero-shot generalization test.
- **Real-world data:** no additional training data — the same Level-I-trained policies are deployed directly on a Franka Emika Panda across five tasks (bowl-to-plate, apple selection, bread-to-toaster, block stacking, dual-arm stacking), each with a physical box obstacle placed along the motion path.

No discussion is given of how representative SafeLIBERO's obstacle placements are of real deployment hazards beyond the two tested levels, nor of any biases in the underlying LIBERO demonstration data.

## Training setup

- **Base policy initialization:** all methods (including baselines) start from the same supervised fine-tuning checkpoint — OpenVLA-OFT fine-tuned on SafeLIBERO demonstrations — isolating the RL/safety method as the only variable being compared.
- **World model training:** initialized from Wan2.2, fine-tuned for 5,000 steps at learning rate $1\times10^{-5}$, DiT-only (VAE frozen), static-video augmentation probability 0.05.
- **Reward/cost head training:** each trained for 20 epochs; task head at learning rate $1\times10^{-4}$, safety head at $3\times10^{-4}$ (batch size 64, weight decay $1\times10^{-4}$, BCE-with-logits loss for both).
- **Policy optimization:** safety-constrained GRPO, group size $G=16$, learning rate $2\times10^{-5}$, FSDP + AdamW, clipping ratios 0.2/0.28, no KL penalty in reported runs ($\beta=0$), safety budget $d=0.2$, multiplier learning rate $\alpha_\eta=0.05$, global batch size 8192, gradient clipping at 1.0.
- **Compute:** world-model fine-tuning on A800 GPUs (data-parallel); full policy optimization on 32 A800-80GB GPUs (4 nodes × 8 GPUs); reward/safety-head training is lightweight single-GPU.
- **Checkpoint granularity:** WoVR (baseline) trains one checkpoint per task suite; WMPO (baseline) and SafeDojo train one checkpoint per individual task — a training-budget asymmetry the paper reports but does not fully control for in the comparison.

## Evaluation & results

**Metrics (Section 5.1):** Task Success Rate (TSR), Safe Success Rate (SSR, the stricter metric requiring task completion *and* zero collision), Collision Avoidance Rate (CAR, percent of episodes with no collision regardless of task outcome), Collision Step Count (CSC, average in-contact simulation steps per episode), and Execution Time Steps (ETS, efficiency).

**Baselines:** OpenVLA-OFT (plain SFT, no RL/safety), AEGIS (inference-time CBF-based action filter — representing "control-theoretic" safety), PPO and SafeVLA (model-free RL, SafeVLA using an explicit CMDP formulation), WMPO and WoVR (model-based/world-model RL *without* explicit safety constraints — the closest prior work architecturally).

**Headline numbers (Tables 1–2):**
- SafeLIBERO Level I: SafeDojo gets the best average TSR (64.50%), best average SSR (53.25%, +8.25pp over SafeVLA's 45.00%, the strongest baseline on this metric), and best average ETS (269.41, i.e., most efficient).
- SafeLIBERO Level II (zero-shot generalization, no Level II training data): SafeDojo again leads on TSR (57.00), SSR (49.62, narrowly ahead of SafeVLA's 49.50), and ETS (277.00). SSR drops 3.63pp from Level I to Level II, smaller degradation than some baselines.
- Real-world (Table 8, 5 Franka tasks × 10 trials each): SafeDojo achieves 76.00% average TSR and 70.00% average SSR — both by a wide margin over the next-best baseline (WoVR at 66.00 TSR but only 22.00 SSR — WoVR completes tasks but collides constantly). This TSR–SSR gap comparison is the paper's central qualitative point: several baselines "succeed" at the task while frequently violating safety.

**Ablations (Section 5.4, Figure 4, single task — Spatial Task 0 only):**
- Removing the safety head: SSR drops from 48% to 38% (TSR relatively preserved) — confirms the safety branch is doing real work, not just adding noise.
- Removing the task reward head: TSR/SSR both collapse to 3% — the policy has no goal-directed signal at all without it.
- Replacing constrained GRPO with unconstrained optimization: TSR and SSR both drop substantially (to 40%/34%) — validates the constrained formulation isn't decorative.
- Removing the adaptive Lagrangian multiplier (presumably fixing $\eta$): TSR stays high (68%) but SSR drops to 46% — shows a fixed penalty weight lets the policy get away with more collisions while still completing tasks.
- Input-source ablation: using only *future* predicted states (without the current imagined observation) is worse (52% TSR / 36% SSR) than the default "current observation + proposed action" input; adding future states *on top of* the default doesn't clearly help either (58% TSR / 50% SSR, but at higher ETS) — the authors attribute this to accumulated world-model prediction error making "future" inputs noisier than they're worth.
- $\eta$ sensitivity: performance is not monotonic — TSR/SSR both peak around $\eta=0.2$ (the value used everywhere else in the paper) and degrade at both lower ($\eta=0.1$) and much higher ($\eta=0.6$) values, the latter due to over-conservatism.

All ablations are run on a single task (Spatial Task 0) with a single fixed initial $\eta=0.3$ (except the $\eta$-sensitivity sweep itself, which varies it) — this is a real limitation on how far the ablation conclusions generalize across the benchmark's other 15 tasks.

## Limitations (as stated by the authors)

Stated explicitly in Appendix B:
1. The safety head is trained only on **binary** collision labels, not finer-grained signals like contact force or proximity margins — a genuinely continuous notion of "how unsafe" isn't captured.
2. World-model prediction errors **accumulate over longer rollout horizons**, degrading both task-progress and safety evaluation reliability the further out the imagination extends.
3. Evaluation is confined to **tabletop manipulation with static obstacles**; dynamic obstacles or human presence are explicitly left as open/untested.
4. SafeDojo is a **training-time** safety mechanism only — it provides no hard, provable safety guarantee at inference time, and the authors explicitly suggest pairing it with inference-time safeguards like CBFs (i.e., combining with the very AEGIS-style baseline it's compared against, rather than replacing it) as future work.

The Broader Impact section (Appendix D) additionally flags that learned safety signals may fail unexpectedly outside the training distribution, and warns against "overreliance on learned safety signals without external safeguards" creating false confidence.

## Critical read

**What's actually novel vs. a recombination.** The individual pieces here are each borrowed from adjacent, already-published work: GRPO itself, the Lagrangian-constrained variant of GRPO (explicitly credited to "Constrained GRPO" [41], a 2026 preprint), the CMDP safety formulation (standard, from Altman 1999), and world-model-based VLA post-training in imagination (WMPO, WoVR — cited as directly comparable prior work). The genuinely new contribution, as the paper's own Table 3 argues, is being the first to combine an *anticipatory* (imagination-based, catches problems before they happen) mechanism with a *learned* (not hand-crafted, not simulator-privileged) safety signal simultaneously — prior model-based safe RL work (SafeDreamer, FOSP) is anticipatory but uses hand-crafted or simulator-provided scalar costs, and prior world-model VLA RL work (WMPO, WoVR) is learned but has no safety constraint at all. That's a real, well-supported gap-filling claim rather than an incremental combination for its own sake — Table 3's two-axis positioning is honest about exactly what's new and what isn't.

**Strength of the empirical case.** The main-table comparisons are reasonably controlled (shared SFT checkpoint, same base policy architecture, same benchmark). The real-world numbers are the most persuasive part of the paper precisely because they show a TSR/SSR *decoupling* in the baselines (WoVR: 66% TSR but only 22% SSR) that a reader might otherwise dismiss as a simulation artifact — seeing it hold up physically is meaningful. That said, real-world results are 10 trials per task per method (50 trials/method total) — a small sample size for percentage-point claims, and no confidence intervals or variance are reported anywhere in the paper (simulation or real-world).

**What the ablations don't establish.** All four component ablations and the input-source ablation are run on a single task (Spatial Task 0), with a fixed initial $\eta = 0.3$ that differs from the paper's own default deployment value ($\eta = 0.2$). It's plausible — and not ruled out — that some of these ablation effects (e.g., the "removing the Lagrangian multiplier" result) are sensitive to that specific starting condition or that specific task's obstacle geometry. The claims in the ablation section are stated in general terms ("safety-cost estimation is critical," "adaptive balancing... is necessary") that are broader than the single-task evidence directly supports.

**Open tension the paper is candid about but doesn't resolve.** The paper's own Appendix B limitation #4 — no hard inference-time safety guarantee — sits in some tension with the main narrative, which is framed as an alternative to (rather than a complement to) inference-time CBF filtering (AEGIS). The paper positions AEGIS as a weaker baseline throughout Table 1/2 (AEGIS has the lowest TSR of any method on Level I, likely because CBF-based filtering is overly conservative when it lacks precise geometric/camera calibration for open-world obstacles), yet its own future-work paragraph suggests SafeDojo should ultimately be paired with exactly that kind of mechanism. The value proposition is therefore "better learned prior for what to imagine avoiding," not "a replacement for hard safety guarantees" — a distinction the abstract and introduction don't foreground as clearly as the appendix does.

**Benchmark scope.** SafeLIBERO's obstacle-interference framing (Level I: near the target; Level II: along the path) is a reasonable but narrow operationalization of "safety" — it is entirely about static-obstacle collision avoidance. The paper is explicit about this scope (Appendix B, limitation #3) rather than overclaiming broader safety coverage (e.g., no claims about semantic/instruction-level safety, human proximity, or force-limit violations, which is a different sense of "VLA safety" than e.g. jailbreak or backdoor-oriented surveys use).

## Reference

Kai Tang, Peidong Jia, Zhong Chu, Jixian Wu, Rui Ma, Jiajun Cao, Fangyuan Zhao, Sixiang Chen, Yichen Guo, Xiaowei Chi, Chun-Kai Fan, Kevin Zhang, Jinchang Xu, Fubing Yang, Weishi Mi, Xiaozhu Ju, Jian Tang, Shanghang Zhang. "SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model." arXiv:2606.20698v1 [cs.RO], 15 Jun 2026.
