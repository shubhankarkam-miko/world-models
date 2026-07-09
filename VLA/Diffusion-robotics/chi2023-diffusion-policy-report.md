# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

## TL;DR

Diffusion Policy represents a robot's visuomotor control policy as a conditional denoising diffusion process over **action sequences**, conditioned on recent visual observations. Instead of regressing directly to an action (or a fixed mixture-of-Gaussians / discretized action), the network learns to predict noise — equivalently, the gradient of the action-distribution's score function — and iteratively denoises a sample from Gaussian noise into a coherent sequence of future actions. Across 15 tasks from 4 established manipulation benchmarks (simulated and real, 2–14 DoF, single- and multi-task, rigid and fluid objects), it beats the prior state of the art (LSTM-GMM/BC-RNN, IBC, BET) by an average of 46.9% success rate. The paper's real contribution isn't "diffusion models can represent policies" — several concurrent papers claim that too — it's the specific engineering recipe that makes this work on physical hardware: receding-horizon closed-loop execution, treating observations purely as conditioning rather than part of the modeled joint distribution (for speed and end-to-end vision training), a transformer variant for high-frequency action changes, and an empirical case for position control over the field's default of velocity control.

## Architecture

**General formulation (Fig. 2a).** At control step *t*, the policy consumes the last *T_o* steps of observation history *O_t* and outputs a predicted action sequence *A_t* of length *T_p* (the *prediction horizon*). Of those *T_p* predicted actions, only *T_a* (the *action horizon*) are actually executed before the policy replans — this is the receding-horizon, closed-loop design. The action sequence itself is produced by *K* [iterative denoising steps](#ddpm-ho-et-al-2020) (the DDPM process): start from *A_t^K* sampled from Gaussian noise, and repeatedly apply

A_t^{k-1} = α(A_t^k − γ·ε_θ(O_t, A_t^k, k) + N(0, σ²I))

until reaching the denoised *A_t^0*. The network *ε_θ* is trained to predict the noise that was added to a clean action sequence (Eq. 5), which is mathematically equivalent to predicting the gradient of the action-distribution's energy/[score function](#score-based-generative-modeling-song-and-ermon-2019) (see Critical read and Eq. 8).

**Two interchangeable noise-prediction backbones (the choice of *ε_θ* is independent of the visual encoder):**

- **CNN-based (Fig. 2b).** A 1D temporal CNN adapted from Janner et al.'s "[Diffuser](#diffuser-janner-et-al-2022)" planning network. Observation features *O_t* are injected at every convolutional layer via [FiLM](#film-conditioning-perez-et-al-2018) (Feature-wise Linear Modulation), channel-wise, alongside a similar conditioning signal for the diffusion step *k*. The network predicts only the action trajectory (not a concatenated observation-action trajectory, unlike Diffuser). Works well out-of-the-box with little tuning; its inductive bias toward low-frequency signals makes it degrade on tasks needing fast, sharp action changes (e.g. velocity control).
- **Transformer-based (Fig. 2c).** Adapts the minGPT decoder architecture used in BET. Noisy action tokens *A_t^k* are the input sequence to a causal transformer decoder — each action token attends only to itself and earlier action tokens. The diffusion step *k* is prepended as a sinusoidal-embedding token. Observation embeddings (produced by a shared MLP) are injected via cross-attention in every decoder block. Better on tasks with high task complexity or fast action changes, but more hyperparameter-sensitive and, unusually, doesn't reliably benefit from more layers.
- **Recommendation given in the paper:** start with CNN; move to the transformer only if performance is limited by task complexity or high-rate action change, at the cost of more tuning.

**Visual encoder.** A ResNet-18 (trained from scratch, no ImageNet pretraining in the main results) per camera view; each timestep's image is encoded independently and the per-view, per-timestep embeddings are concatenated into *O_t*. Two modifications from a stock ResNet-18: (1) global average pooling → spatial softmax pooling, to preserve spatial information; (2) BatchNorm → GroupNorm, because BatchNorm's running statistics interact badly with the Exponential Moving Average (EMA) commonly used to stabilize DDPM training.

**Conditioning, not joint modeling — the key speed/trainability decision.** The policy models the *conditional* distribution p(A_t | O_t) rather than a *joint* distribution p(A_t, O_t), which is what Janner et al.'s diffusion-for-planning work does. Concretely, the observation is never part of the noised/denoised tensor — it's injected as a conditioning signal at every denoising step (Eq. 4, Eq. 5). This means the vision encoder is run once per control step instead of once per denoising iteration, which is both the main reason inference is fast enough for real-time control and the reason end-to-end training of the vision encoder is tractable at all.

**Inference acceleration.** [DDIM](#ddim-song-et-al-2021) decouples the number of training vs. inference denoising iterations. The paper trains with 100 iterations everywhere but runs inference with 100 (simulation) or 16 (real robot) iterations, hitting 0.1s latency on an RTX 3080 in the real-world setup.

**Noise schedule.** The Square Cosine schedule from [iDDPM](#iddpm-cosine-noise-schedule-nichol-and-dhariwal-2021) was found empirically to work best for their control tasks; the paper does not derive why, only reports the empirical outcome.

## Data

Fifteen tasks drawn from four existing benchmarks, deliberately chosen to span axes that behavior-cloning papers often don't stress-test together: state vs. image observation, 2-DoF to 6-DoF (14-DoF for the two-robot Transport task) action spaces, single- vs. multi-task settings, fully- vs. under-actuated systems, rigid vs. fluid/deformable objects, and demonstrations from single vs. multiple human operators.

- **Robomimic** (Mandlekar et al. 2021): Lift, Can, Square, Transport, ToolHang — proficient-human (PH) demo sets for all 5, plus mixed proficient/non-proficient (MH) sets for 4 of them (200 PH / 300 MH demos typically).
- **Push-T** (adapted from Florence et al.'s IBC): pushing a T-block to a fixed target; two observation variants (RGB image, or 9 ground-truth 2D keypoints), 300 scripted-style demonstration episodes.
- **Multimodal Block Pushing** (adapted from BET / Shafiullah et al.): pushing two blocks into two target squares in *arbitrary order* — a long-horizon multimodality stress test. Demonstrations here are **not human** — 1000 episodes generated by a scripted oracle with ground-truth state access. This is a meaningfully different data-generating process from the rest of the benchmark and the paper explicitly notes the optimal observation/action horizon for this task differs from the human-teleop tasks.
- **Franka Kitchen** (Gupta et al., Relay Policy Learning): 656 human demonstrations, each completing 4 of 7 possible sub-tasks in arbitrary order — short- and long-horizon multimodality.
- **Real-world tasks:** Push-T (136 demos, single proficient operator, "many hours" of prior practice), 6DoF sauce pouring and periodic spreading (50 demos each, 90% used for training), mug flipping (250 demos), plus three bimanual tasks added in this extended version — egg beating (210 demos), mat unrolling (162 demos), shirt folding (284 demos).

**Preprocessing notes worth flagging (Appendix A):** action dimensions are min-max normalized per-dimension to [−1, 1] rather than zero-mean/unit-variance, because DDPM sampling clips to [−1, 1] at every iteration — z-scoring would leave parts of the action space permanently unreachable. Near-constant action dimensions are zero-centered without scaling to avoid numerical blow-up. Rotation dimensions are left unnormalized. Velocity-control tasks use axis-angle rotation representation; all position-control tasks (sim and real) use the [6D rotation representation](#6d-rotation-representation-zhou-et-al-2019) from Zhou et al. (2019) instead.

## Training setup

- **Objective:** standard DDPM noise-prediction MSE loss, conditioned on observations (Eq. 5): sample a clean action chunk, a random diffusion step *k*, add noise appropriate to that step, and train *ε_θ* to predict the added noise.
- **Diffusion iterations:** 100 for training, both architectures, both simulation and real-world. Inference uses DDIM with the same 100 steps in simulation, reduced to 16 on real hardware for latency.
- **Batch size:** 256 for all state-based experiments, 64 for all image-based experiments.
- **LR schedule:** cosine with linear warmup — 500 steps for the CNN backbone, 1000 steps for the transformer backbone.
- **Epochs:** 4500 for state-based tasks, 3000 for image-based tasks (simulation); real-world Push-T used a fixed 12-hour training budget per method instead of a fixed epoch count.
- **Weight decay** differs sharply by backbone: ~1e-6 for the CNN model vs. 1e-3 (1e-1 for Push-T specifically) for the transformer — consistent with the paper's observation that the transformer variant is considerably more hyperparameter-sensitive.
- **Checkpoint selection:** reported as (best single checkpoint) / (average of last 10 checkpoints, saved every 50 epochs) across 3 training seeds × 50 environment initial conditions — an explicit attempt to avoid the "cherry-pick the best checkpoint" failure mode the paper documents for IBC (Sec. 4.4).
- **Action space:** position control used for all Diffusion Policy results; baselines are each evaluated at their own best-performing action space (velocity control, in every baseline's case) — i.e., every method gets to use its own favorite action space rather than being forced onto a common one. This is evaluation-methodology-relevant, see Critical read.
- **Image augmentation:** random crop, following Mandlekar et al.'s Robomimic recipe; crop sizes are task-specific (Table 7 in the appendix).

## Evaluation & results

**Benchmarks and metrics.** Success rate for nearly every task; Push-T uses target-area IoU/coverage instead, since "success" there is graded. Baselines: LSTM-GMM (the paper's own reproduction of BC-RNN, which came out slightly better than the original Robomimic paper's numbers), IBC, and BET — each reported using whichever of {the paper's own reproduction, the original paper's number} was strongest for that baseline.

**Headline result:** Diffusion Policy variants win on all 15 tasks/variants in both state-based and image-based observation settings, with an average success-rate improvement of 46.9%, computed per-task as (best Diffusion Policy variant − best baseline) / best baseline, then averaged unweighted across tasks (Appendix B.2). Selected concrete numbers: on Robomimic Transport (mh), DiffusionPolicy-C reaches 0.68 vs. baselines' best of 0.62 (BET); on ToolHang, DiffusionPolicy-T reaches 1.00 vs. 0.67 (LSTM-GMM); on Kitchen's hardest multi-object metric (p4), 0.99 vs. 0.44 — a 213% relative improvement the paper calls out explicitly as its long-horizon-multimodality result. On Multimodal Block Pushing's hardest metric (p2, both blocks pushed), DiffusionPolicy-T gets 0.94 vs. BET's 0.71.

**Real-world results:** Push-T 95% success (vs. 20% LSTM-GMM, 0% IBC) at 0.80 IoU vs. 0.84 for human demonstrations; mug flipping 90% vs. 0% for LSTM-GMM; sauce pouring/spreading close to human-level (0.74 vs. 0.79 IoU on pouring, full success on spreading); the new bimanual tasks land lower — egg beating 55%, mat unrolling 75%, shirt folding 75% — with the paper attributing most bimanual failures to missed/lost grasps rather than diffusion-specific issues.

**Ablations worth knowing about:**
- *Action horizon:* trades off temporal consistency against reaction latency; ~8 steps was best across most tasks tested — the curve degrades sharply past that.
- *Latency robustness:* Diffusion Policy with position control holds peak performance up to ~4 steps of simulated latency; velocity control degrades faster, attributed to compounding error.
- *Position vs. velocity control:* switching Diffusion Policy from velocity to position control *improves* it, while the same switch *hurts* LSTM-GMM and BET — this is presented as one of the paper's more counter-to-literature findings (most prior behavior-cloning work defaults to velocity control).
- *Vision encoder pretraining (Table 5, single task — Robomimic Square PH only):* fine-tuning a pretrained encoder at 10x lower LR than the policy beat both training from scratch and using a frozen pretrained encoder; frozen encoders performed *worse* than scratch training for ResNet, which the paper reads as evidence that diffusion policy wants a different visual representation than standard pretraining produces. CLIP-pretrained ViT-B/16 with fine-tuning reached 98% in only 50 epochs, the single best number in the table.
- *Data efficiency:* Diffusion Policy beats LSTM-GMM at every training-set size tested (40–200 demonstration episodes on Push-T and Square).
- *Observation horizon:* state-based policies are insensitive to it; vision-based policies (especially the CNN backbone) degrade as it grows, with 2 steps being a reasonable default for both.

## Limitations (as stated by the authors)

The paper is fairly candid and limits itself to two explicitly stated limitations (Sec. 9):

1. **Inherits standard behavior-cloning limitations** — performance is capped by demonstration data adequacy; the authors note the framework could in principle be extended to reinforcement-learning-style training to exploit suboptimal/negative data, but don't do so here.
2. **Higher computational cost and inference latency** than simpler methods like LSTM-GMM. Action-sequence prediction partially compensates (since replanning happens less often), but the paper states this may still not suffice for tasks requiring genuinely high-rate control, and points to diffusion-acceleration techniques (better noise schedules, faster solvers, consistency models) as future work rather than something they've solved.

No other limitations are claimed by the authors in that section; anything else below is this report's own read, not theirs.

## Critical read

**The "novel contribution" is narrower than the abstract implies, and the paper is honest about this if you read past the abstract.** Diffusion models as a policy class predate this paper — Wang et al. (2022) proposed diffusion policies for offline RL earlier, and the paper explicitly flags three concurrent works (Pearce et al. 2023, Reuss et al. 2023, Hansen-Estruch et al. 2023) doing closely related things at the same time, differing mainly in emphasis (sampling strategy and RL applications for them, action-space and real-hardware engineering for this paper). [Certain, stated directly by the authors in Sec. 8] The real contribution is the systems-engineering bundle: receding-horizon closed-loop execution + observation-as-conditioning (not joint modeling) + the position-control finding + a working transformer variant, validated unusually thoroughly across 15 tasks and real bimanual hardware. That's a legitimate and useful contribution — but it's an engineering-and-evaluation paper wearing an "introduces a new way of generating robot behavior" abstract, more than it's a fundamentally new modeling idea.

**The headline 46.9% number is a specific, somewhat generous construction, not a neutral summary statistic.** It's computed as an *unweighted mean of per-task relative improvements*, using each task's *strongest* baseline as the denominator (Appendix B.2). Two things follow: (a) tasks where even the best baseline is weak contribute disproportionately to the average in percentage terms — e.g., Kitchen's p4 metric goes from 0.44 to 0.99, a 213% "improvement" that would look far less dramatic as an absolute 55-point gain; (b) because each baseline method is also allowed to run at its own best action space (velocity), while Diffusion Policy runs at position control, the comparison is "best version of X vs. best version of Diffusion Policy," not "same action space, network changed" — which is a defensible methodological choice (the paper argues position control is *part of* what makes Diffusion Policy work) but conflates two independent variables (policy architecture, and action-space choice) into a single headline number. Worth remembering when someone quotes "46.9% better" without the caveats.

**Position-control result is compelling but tested on a fairly narrow slice of the benchmark suite.** The velocity-vs-position ablation (Fig. 4) is shown on exactly two tasks — Square and Kitchen p4. It's a real, interesting finding, but "consistently outperforms" (the paper's phrasing in Sec 4.2) is a stronger claim than two data points fully support; it reads as a plausible general trend that the paper is confident about rather than one that's been broadly stress-tested within this paper itself.

**The multimodality story is qualitative for the most compelling case.** The flagship demonstration of multimodal expressiveness (Fig. 3, Push-T left/right push) is a single qualitative visualization of rollout trajectories, not a quantified metric — contrast with the Block Push p1/p2 and Kitchen p1–p4 numbers, which are real quantitative multimodality metrics. The paper is not misleading about this (it presents Fig. 3 as an illustration, not a benchmark row), but a reader skimming for "does diffusion really solve multimodality" should look at Table 4, not Figure 3, for the actual evidence.

**Bimanual results are the honest weak spot, and appropriately so.** 55% on egg beating and 75% on the other two bimanual tasks are real limitations of the current system, openly reported with failure-mode breakdowns (missed grasps, inability to self-terminate cleanly on shirt folding). These numbers are presented straightforwardly rather than being spun, which is a point in the paper's favor even though the numbers themselves are unremarkable.

## Reference

Chi C, Xu Z, Feng S, Cousineau E, Du Y, Burchfiel B, Tedrake R, Song S (2024). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* The International Journal of Robotics Research (extended version of the RSS 2023 conference paper: Chi C, Feng S, Du Y, Xu Z, Cousineau E, Burchfiel B, Song S (2023), *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, Proceedings of Robotics: Science and Systems). Project page: diffusion-policy.cs.columbia.edu

## Background: Borrowed Concepts

Citations the paper leans on as building blocks without re-explaining — flagged by the reference-lookout pass, approved for write-up. Each entry states confidence plainly; these are glossary-style explanations to unblock the citing sentence, not full paper summaries.

### DDPM (Ho et al., 2020)
[Certain] DDPMs define a fixed forward process that gradually adds Gaussian noise to data over many steps until it's indistinguishable from pure noise, then train a neural network to reverse that process one step at a time. The trick that makes this tractable: instead of training the network to directly predict the denoised data, it's trained to predict the *noise* that was added at each step — a much easier regression target. Ho et al. show that minimizing this simple noise-prediction MSE loss is mathematically equivalent, up to a reweighting, to minimizing a variational lower bound on the log-likelihood of the reverse process — the theoretical justification for why "just predict the noise" produces a valid generative model rather than an ad hoc heuristic. Diffusion Policy inherits this loss and this justification wholesale, applying it to action sequences instead of images.

[↩ back to Architecture](#architecture)

### Score-Based Generative Modeling (Song and Ermon, 2019)
[Certain on the core idea; general rather than paper-specific familiarity] This work frames generative modeling as learning the score function — the gradient of the log-density, ∇ₓ log p(x) — at every point in data space, rather than the density itself. Once you have that gradient field, you can sample from the distribution via Langevin dynamics: repeatedly step in the direction the score points, with a bit of extra noise added at each step to keep exploring rather than collapsing onto a single mode. The paper's practical contribution was showing a network can be trained to estimate this score directly, and that using multiple noise levels during training (rather than one) stabilizes it dramatically. Diffusion Policy's claim that ε_θ "approximates the negative score function" (Eq. 8) is this framework transplanted onto p(actions | observations): the noise-prediction network doubles as a score estimator, which is what licenses the "iteratively walk toward higher-probability actions" reading of the denoising loop.

[↩ back to Architecture](#architecture)

### FiLM Conditioning (Perez et al., 2018)
[Certain] FiLM (Feature-wise Linear Modulation) injects side information into a convolutional network without concatenating it into the input. At each layer, a small side network maps the conditioning signal — here, the observation embedding — to a per-channel scale γ and shift β, and every channel of that layer's feature map is transformed as γ·x + β before continuing through the network. It's cheap, applies uniformly at every layer, and — unlike concatenating the condition once at the input — re-injects the conditioning signal's influence at every depth, which matters for a diffusion model that needs the observation to steer denoising consistently across many layers. Diffusion Policy uses this exact mechanism to fuse O_t and the diffusion-step embedding *k* into the CNN backbone at every conv layer.

[↩ back to Architecture](#architecture)

### Diffuser (Janner et al., 2022)
[Likely — confident on the architecture and framing; less certain on exact layer-by-layer specifics of their U-Net] Diffuser trains a diffusion model over entire state-action trajectories, not single steps, using a 1D temporal convolutional U-Net where the "image" being denoised is a trajectory laid out as a 2D array (time × state-action dimensions). Planning becomes sampling: draw a clean trajectory from the diffusion model, optionally guided toward high reward, and execute it. Diffusion Policy borrows this 1D temporal-convolution backbone directly but changes what's inside the array — actions only, never states/observations — and changes what steers the sampling — FiLM observation-conditioning rather than joint generation or reward-guidance. The exact U-Net layer structure (kernel sizes, up/downsampling depth) lives in Janner et al. directly; this paper doesn't reproduce it.

[↩ back to Architecture](#architecture)

### iDDPM Cosine Noise Schedule (Nichol and Dhariwal, 2021)
[Certain on the general mechanism; Likely on exact schedule-formula recall] A noise schedule decides how much noise gets added at each of the *K* forward diffusion steps — too fast and the model has too little signal to work with near the end; too slow and steps get wasted doing almost nothing. Ho et al.'s original DDPM used a linear schedule; Nichol & Dhariwal showed a cosine-shaped schedule (noise ramps up slowly at the start and end, faster in the middle) preserves useful signal for longer and gives better sample quality at low step counts — directly relevant to Diffusion Policy needing to run inference at real-time robot control rates. The "square cosine" variant used here is a specific parameterization within this family; the reason the shape matters is this signal-preservation argument.

[↩ back to Architecture](#architecture)

### 6D Rotation Representation (Zhou et al., 2019)
[Certain] Common rotation representations — Euler angles, quaternions, axis-angle — all have a discontinuity or non-uniqueness problem somewhere on the space of rotations. That's fine for storage but bad for a network regressing rotations continuously, since it ends up having to learn a jump discontinuity that isn't actually present in the underlying rotation space. Zhou et al. propose representing a rotation with the first two columns (6 numbers) of its 3×3 rotation matrix, then reconstructing a valid orthonormal matrix from those 6 numbers via Gram-Schmidt orthogonalization at use-time. This representation is continuous everywhere, which empirically makes it much easier for a network to regress accurately. Diffusion Policy uses it for every position-control action space instead of axis-angle, which it reserves for velocity-control tasks where rotation commands stay close to zero and the discontinuity rarely bites.

[↩ back to Data](#data)

### DDIM (Song et al., 2021)
[Certain] DDIM reformulates DDPM sampling as a non-Markovian, deterministic mapping between noise and data, with two consequences that matter here: sampling no longer requires walking through every training-time noise level one at a time — steps can be skipped while still landing close to the trained model's actual output — and, being deterministic, the same starting noise reliably produces the same output, unlike the stochastic Langevin-style sampling in vanilla DDPM. This is the exact lever Diffusion Policy pulls to compress inference from 100 training-time denoising steps down to 16 or fewer on real hardware, without training a separate fast model.

[↩ back to Architecture](#architecture)
