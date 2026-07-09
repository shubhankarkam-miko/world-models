# Real-Time Inference for Diffusion-Based Robot Control: Muninn vs. FRMD

*Combined deep-dive covering two papers that attack the same bottleneck — iterative denoising makes diffusion-based trajectory/action generation too slow for real-time robot control — from opposite ends of the training-cost spectrum.*

## TL;DR

**Muninn** (Puthumanaillam, Jiang, Hernandez, Fuentes, Padrao, Bobadilla & Ornik) is a *training-free* wrapper: it wraps an already-trained diffusion trajectory planner and, at each denoising step, decides whether to reuse a cached denoiser output or recompute it, using a cheap internal-representation probe calibrated (via split-conformal prediction) into a statistically certified bound on how far the resulting trajectory can deviate from the full-compute one. **FRMD** (Shi, Hu & Jin) is a *retrain-into-one-step* approach: it distills a pretrained multi-step diffusion policy into a single-step "consistency" student, and — specifically to make that distillation work well for robot control — does the diffusion in a low-dimensional movement-primitive parameter space rather than raw action space. Muninn needs no retraining and is architecture-agnostic; FRMD needs a full distillation pipeline and a committed trajectory parameterization but removes the iterative sampler altogether. Read together, they map out the two ends of a real trade space: certified, drop-in, training-free acceleration vs. maximal, one-time-cost speedup.

---

## Paper 1 — Muninn: Your Trajectory Diffusion Model But Faster

### Architecture

Muninn does not introduce a new denoiser or sampler — it's a decision policy wrapped around whatever pretrained diffusion planner you already have, `τ_{t-1} = Φ_t(τ_t, ε_θ(τ_t, t, c), ξ_t)` for `t = T,…,1`. Its whole design rests on two things every such planner already exposes:

- **A cheap probe of internal stability.** Any modern denoiser evaluation decomposes into a cheap "stem" (input processing that turns `(τ_t, t, c)` into an intermediate representation `h_t`) and an expensive "core" (the deep blocks that turn `h_t` into the final noise prediction). Muninn taps only the stem to compute a probe feature `F_t`, and defines a per-step **score** `s_t := ||F_t − F_{t+1}||₁ / (||F_{t+1}||₁ + ω)` — how much the probe changed since the previous diffusion step. Small change → the denoiser output is likely stable → safe to reuse. Large change → recompute.
- **The sampler's known analytic sensitivity.** For DDPM/DDIM-style samplers, `Φ_t` is affine, so a local Lipschitz bound `‖Φ_t(τ,ε,ξ) − Φ_t(τ′,ε′,ξ)‖ ≤ K_t‖τ−τ′‖ + L′_t‖ε−ε′‖` is available in closed form. Unrolling this recursion from `t=T` down to `t=0` gives a closed-form bound on the final trajectory deviation: an error injected at step `t` gets amplified by a pathwise sensitivity coefficient `L_t` that depends on *every later step's* sensitivity — early errors can matter far more than late ones.

Putting these together: the score `s_t` says *how risky reuse looks right now*; the Lipschitz bound says *how costly that risk would be if realized*. Muninn connects the two with **split-conformal calibration**: on offline calibration episodes, it runs the full-compute chain alongside a "ghost" reuse chain, collects (score, realized-error) pairs, fits a regressor, and adds a conformal residual quantile to get `U(s)` — a distribution-free, high-probability upper bound on the reuse error at a given score. At deployment, this becomes **Algorithm 1**: maintain a remaining deviation budget `B_rem`, and at each step either recompute (if the upper-bounded cost of reusing would exceed the remaining budget) or reuse and spend the budget. A union bound over per-step coverage turns the per-step guarantees into one global guarantee: `P(trajectory deviation > η_traj) ≤ α`, for user-chosen tolerance `η_traj` and risk level `α`.

Practical engineering choices: reuse is forbidden in a short high-noise prefix and low-noise suffix of the diffusion chain; batched sampling tracks an independent budget per trajectory; and the probe definition is architecture-specific but always cheap (a short prefix of attention/MLP blocks mean-pooled for Transformer denoisers, the penultimate hidden layer for MLP denoisers).

### Data

Muninn doesn't train a model — it *calibrates* against episodes drawn from each benchmark's own deployment distribution, then wraps already-published pretrained planners: the D4RL suite (MuJoCo locomotion — HalfCheetah/Hopper/Walker2d; goal-reaching Maze2D/AntMaze; long-horizon FrankaKitchen manipulation; Kuka block stacking), configuration-space arm planning under the EDMP/MPD clutter protocol, and visuomotor diffusion policies (RLBench Reach Target, Meta-World pick-place-v2, DP3 Pour). It further validates on three real hardware platforms: a SeaRobotics Surveyor ASV (marine, waypoint following), a Crazyflie quadrotor (aerial, 3D navigation), and a SO-ARM100 tabletop arm (pick-and-place, stacking, peg insertion).

### Training setup

There is no gradient training for Muninn itself. The only fitting step is the conformal calibration: `N` i.i.d. calibration episodes → score–error pairs → regressor `m(s)` fit on a training split → split-conformal residual quantile `q_{1−α_step}` from a held-out split → `U(s) := max(m(s) + q_{1−α_step}, 0)`. Two user-facing knobs: deviation tolerance `η_traj` and risk level `α` (with `α_step := α / |T_cache|`, chosen so the union bound over cached timesteps gives exactly `α` global risk). All experiments run on a single NVIDIA A10 GPU (24GB), evaluated over 150 held-out episodes in simulation and 150 runs on hardware.

### Evaluation & results

- **Offline RL / trajectory planning (Table I):** wrapping Diffuser, Decision Diffuser, Diffusion-QL, AdaptDiffuser, and CompDiff with Muninn preserves the task metric almost exactly (e.g., Diffuser on HalfCheetah: 59.3 → 58.8) while cutting denoiser evaluations roughly 3–4× (e.g., Diffuser on Maze2D: 733 → 175 evaluations), with bounded, reported trajectory deviation and violation probability.
- **Configuration-space motion planning (Table II)** and **visuomotor imitation (Table III):** success/collision rates stay within about a point of the full-compute model while latency drops 30–50%.
- **Hardware deployment (Table IV):** on the ASV, UAV, and SO-100 arm, Muninn roughly halves to a third of the teacher's latency (e.g., GC-Diffuser on the ASV: 70 ms → 28 ms) with success and collision rates within ~1 point of full compute.
- **Vs. other training-free accelerations (KF#1, Fig. 8):** against FewSteps (shorter horizon), FixedSkip (periodic reuse), and ProbeThresh (probe-threshold reuse with no calibrated bound), Muninn dominates the latency/task-performance Pareto frontier — e.g., reaching full-model success on D4RL AntMaze at roughly half the latency of the next-best baseline. The authors attribute this to being the only method that both estimates reuse error with distribution-free coverage *and* accounts for step-dependent error amplification via `L_t`.
- **Vs. training-time accelerations (KF#2, Fig. 9):** against distilled (1-step/K-step) planners, a smaller from-scratch network, and a learned early-exit controller, Muninn — despite needing zero extra training — recovers a large fraction of the achievable speedup, and *composes*: stacking Muninn on top of an already-distilled model yields further speedup ("Distill+Muninn").
- **Difficulty-adaptive compute (KF#3, Table VII):** Muninn's reuse decisions are context-sensitive — harder, tighter-clearance episodes automatically get more recomputation; easier ones get more reuse.
- **Runtime certificate (KF#4, Fig. 10):** the predicted deviation budget is empirically well-calibrated against realized deviation, and on hardware, spikes in the tracked budget align with near-collision/contact events — the uncertainty signal itself carries safety-relevant information, not just a speed knob.

### Limitations (as stated by the authors)

- The method is explicitly scoped to *state-space trajectory diffusion models*; extending it to other diffusion families "mainly requires redefining the probe... though the adaptation can be nontrivial."
- The deviation budget `η_traj` is fixed and user-specified for a whole deployment; the authors flag as future work replacing it with a *context-aware* budget that "automatically tightens in cluttered or contact-rich regimes and relaxes in open-space or low-risk regimes."
- The formal risk guarantee rests on an **exchangeability assumption** between calibration and deployment episodes — the paper states this as an assumption rather than a limitation, but it means the certificate is only as good as how representative the calibration episodes are of actual deployment conditions; a distribution shift after calibration isn't covered.

---

## Paper 2 — FRMD: Fast Robot Motion Diffusion via Trajectory-Level Consistency Distillation

### Architecture

FRMD targets the same bottleneck from the opposite direction: instead of accelerating an existing sampler, it retrains a *replacement* for it. Two ideas are combined:

**1. Diffuse in movement-primitive space, not action space.** Rather than denoising a raw trajectory `τ ∈ ℝ^{n×k}`, FRMD denoises a compact weight vector `w ∈ ℝ^{kd}` that parameterizes a **Probabilistic Dynamic Movement Primitive (ProDMP)**: `y(t) = c₁(y₀,ẏ₀)ψ₁(t) + c₂(y₀,ẏ₀)ψ₂(t) + Φ(t)ᵀw`, where `Φ(t)` is a fixed bank of basis functions and `c₁, c₂` are determined by boundary conditions. A deterministic decoder `G` expands `w` (+ boundary conditions) into the full action trajectory. The **teacher model** `F_θ` is an encoder that maps `(noisy trajectory, observation, diffusion time)` to denoised MP weights, followed by this fixed ProDMP decoder — trained with standard denoising score matching under the probability-flow ODE (score approximated via Tweedie's formula), and sampled at inference by numerically integrating the ODE with a high-order solver (DPM-Solver).

**2. Collapse the teacher's multi-step sampling into one step via consistency distillation.** The **student** is a consistency function `f_θ(τ,o,s) = c_skip(s)·τ + c_out(s)·F_θ(τ,o,s)`, with the skip/output coefficients chosen so that `f_θ(τ₀,0) = τ₀` — clean trajectories are fixed points of the mapping, by construction. Training uses a student–target setup: both networks are initialized from the pretrained teacher; the student predicts a clean trajectory directly from a very noisy input, while a slowly EMA-updated target network predicts from a teacher-denoised (`K`-ODE-step) intermediate state along the *same* probability-flow-ODE trajectory; the student is trained to match the target's prediction (the trajectory-level consistency distillation loss). At inference, only the student survives: one forward pass, `τ₀ = f_θ(τ_S, o, σ_max)`, no ODE solver needed.

### Data

12 manipulation tasks from **MetaWorld** and **ManiSkill** (4 easy, 5 medium, 3 hard — e.g., PickCube-v1/Reach-v1 easy; PushT-v1/Door-v1 medium; PegInsertion-v1/PlugCharger-v1 hard), 100 expert demonstrations per task (pretrained policies for MetaWorld, trajectory replay for ManiSkill). Demonstrations are sliced with a sliding window into (3-frame observation history) → (12-step action horizon) samples. RGB observations go through a modified ResNet-18 with spatial-softmax pooling and GroupNorm.

### Training setup

Two stages, both AdamW, batch 128, lr 1e-4, weight decay 1e-6, 30k steps, cosine schedule with 500 warmup steps, single RTX 4090: (1) train the teacher diffusion-in-MP-space model via denoising score matching; (2) distill it into the student via trajectory-level consistency distillation, using `K=2` teacher ODE steps per distillation target and EMA decay `μ=0.95`. Backbone for MPD/FRMD is a 6-layer, 4-head Transformer (hidden dim 256, ~8.4M parameters); the ProDMP decoder uses `d=20` Gaussian radial basis functions per degree of freedom, fixed offline.

### Evaluation & results

- **Headline (Table 3):** FRMD reaches 64.8% overall success — edging out its own teacher MPD (64.1%) and clearly beating vanilla Diffusion Policy (50.1%) — at **17.2 ms** average inference, vs. 168.6 ms for the MPD teacher (**~10×**) and 119.7 ms for Diffusion Policy (**~7×**).
- **By difficulty:** on easy tasks all three methods saturate (>98%), but FRMD is still ~8× faster than Diffusion Policy at equal success. On medium tasks, FRMD (66.3%) > MPD (64.8%) > DP (41.0%). On hard tasks, FRMD (29.0%) > MPD (28.6%) ≫ DP (10.1%) — and FRMD's latency stays flat (~18 ms) across difficulty levels while DP's success collapses as difficulty rises.
- **Motion smoothness:** on PlugCharger-v1, FRMD produces 21 non-smooth (high-curvature) transitions vs. 82 for Diffusion Policy — a 74% reduction, which the authors attribute to the built-in smoothness of the ProDMP basis expansion.
- **Learning dynamics:** FRMD initially trails the teacher (adapting to single-step generation) for the first 5–10k steps, then overtakes both baselines; on hard tasks it converges faster than the teacher (~20k vs. ~25k steps) with lower cross-seed variance, which the authors attribute to the combination of the structured MP action space and the EMA-stabilized consistency targets.
- **Backbone ablation (Table 4):** a lighter Transformer-Lite (3 layers) slightly *improves* medium-task success while cutting latency; an MLP backbone is fastest (~9–10 ms) but success on hard tasks collapses to 9.2% — capacity still matters on the hardest tasks even after distillation.
- Headline framing: this puts effective control frequency at **~58 Hz**, vs. 6–8 Hz for the multi-step diffusion baselines it compares against.

### Limitations (as stated by the authors — Section 4.4)

1. **Applicability to predictable environments.** The method is motivated by online generation under uncertainty; in fully predictable settings where accurate open-loop trajectories can be precomputed, the advantage may shrink, and the paper doesn't run that comparison.
2. **Simulation-only evaluation.** Reported latency excludes real sensing/communication/actuation delays; real-world sensor noise and execution delays could reduce the practical benefit. Physical hardware validation is explicitly left to future work.
3. **Narrow motion-quality analysis.** The smoothness comparison is done on one representative medium task (PlugCharger-v1) rather than across the full 12-task suite, without statistical-significance testing.
4. **Fixed, short trajectory horizon.** Only short, fixed-horizon trajectories are evaluated; longer, compositional, or dynamically re-planned trajectories would likely need extra machinery (hierarchy, horizon adaptation) not built here.
5. **Offline, hand-set ProDMP parameterization.** The number and placement of basis functions is fixed via standard practice rather than optimized per task; this design space is unexplored.
6. **No end-to-end VLA integration tested.** FRMD is proposed as a modular action-decoder for vision-language-action systems but was never actually plugged into one in this work.

---

## Comparison: two paths to real-time diffusion inference

Both papers target exactly the same symptom — diffusion-based trajectory/action generation is too slow for real-time robot control because of iterative denoising — but sit at opposite ends of a training-cost spectrum that Muninn's own paper explicitly plots itself against (its Fig. 9 comparison of inference-time vs. training-time acceleration methods):

| | **Muninn** | **FRMD** |
|---|---|---|
| Retraining required | None — training-free wrapper | Full two-stage pipeline (train teacher, then distill student) |
| What's accelerated | Denoiser *evaluations* within an unmodified sampler (skips redundant network calls) | The sampler *itself* (collapses ~10–100 ODE/denoising steps into 1 forward pass) |
| Architecture constraint | Works on any state-space trajectory diffusion model (probe is architecture-specific but the framework is not) | Commits to a specific structured trajectory parameterization (ProDMP) and a consistency-model backbone |
| Speedup achieved | Up to 4.6× wall-clock, 7.7× fewer denoiser calls | ~10× vs. its own teacher, ~7× vs. vanilla Diffusion Policy |
| Safety/quality accounting | Formal, distribution-free statistical guarantee (split-conformal) on trajectory deviation | Empirical — matches/exceeds teacher task success and smoothness, no formal deviation bound |
| Compute allocation | Context-adaptive per episode (harder situations get more recompute) | Fixed — single forward pass regardless of difficulty |

The two are not actually competitors so much as complementary points on the same axis. Muninn's own experiments (Table VI, "Distill+Muninn") show that its training-free caching *composes* with a distilled model — applying Muninn's step-skipping on top of an already-distilled planner still yields further speedup, because Muninn targets redundant per-step computation that survives even in a fewer-step sampler. FRMD's student is already single-step, so that specific composition wouldn't apply as-is, but the underlying idea generalizes: FRMD's aggressive one-time cost buys the largest possible speedup and needs no runtime accounting, while Muninn buys a smaller-but-still-substantial, statistically certified speedup on a model you don't want to (or can't) retrain — e.g., a third-party pretrained planner, or a setting where you need an auditable bound on control-relevant deviation rather than just "it still works empirically."

## Critical read

**Muninn's real contribution** is bringing an actual distribution-free statistical guarantee — split-conformal prediction, an established tool borrowed wholesale from the conformal-prediction literature — to bear on diffusion caching, and mapping that guarantee explicitly onto *trajectory deviation* rather than a perceptual/pixel-space proxy. The underlying mechanism (reuse a cached denoiser output across sampler steps when an internal signal looks stable) is not new to diffusion acceleration in general — the paper's own related work cites several prior training-free caching methods for image/video diffusion (DeepCache, FasterCache, etc.) that do something structurally similar. What's genuinely new is the calibration layer: turning "this looks stable" into "we can certify, with probability ≥ 1−α, that the final trajectory won't deviate by more than η_traj" — and doing so without any distributional assumption beyond exchangeability between calibration and deployment episodes. That's a meaningfully different (and more useful, for control) claim than the perceptual-similarity claims typical of prior work in this space.

**FRMD's real contribution** is narrower and more incremental: both of its ingredients — consistency distillation (Song et al., 2023) and ProDMPs — are directly imported, and the paper's own comparison table cites directly analogous prior work (Consistency Policy, ManiCM) that already applies consistency distillation to diffusion-based robot policies. The specific new piece is doing that distillation *in ProDMP weight space* rather than raw action space, and the paper's smoothness/convergence results suggest this matters. However, the paper never isolates that claim cleanly: there's no ablation comparing "consistency-distilled Diffusion Policy in raw action space" against FRMD, so the reported gains over Diffusion Policy (50.1%) conflate two changes at once (distillation *and* the structured trajectory space). The comparison against MPD (which has the ProDMP space but not distillation) partially isolates the distillation contribution, but the symmetric ablation — distillation without ProDMP — is missing, which is a real gap in an otherwise clean experimental writeup, especially given the six limitations the authors themselves are candid about (simulation-only, narrow smoothness analysis, no VLA integration, etc.).

**Where they leave the reader:** together, the two papers make a reasonably convincing case that the ~10–100 step iterative sampling loop in diffusion-based robot control is *not* fundamental — it can be cut by roughly an order of magnitude either without touching the model (Muninn) or by committing to retrain it once (FRMD) — but neither paper benchmarks against the other's approach directly, so a reader is left to reconstruct the trade-off (as above) rather than finding it adjudicated experimentally in either paper.

## References

Puthumanaillam, G., Jiang, H., Hernandez, R., Fuentes, J., Padrao, P., Bobadilla, L., and Ornik, M. **Muninn: Your Trajectory Diffusion Model But Faster.** arXiv:2605.09999v1 [cs.RO], 11 May 2026. Code: https://github.com/gokulp01/Muninn

Shi, X., Hu, Y., and Jin, J. **FRMD: Fast Robot Motion Diffusion via Trajectory-Level Consistency Distillation.** Frontiers in Robotics and AI 13:1751688, 2026. doi:10.3389/frobt.2026.1751688
