# Planning with Diffusion for Flexible Behavior Synthesis

*Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine — ICML 2022, arXiv:2205.09991 ("Diffuser")*

## TL;DR

The paper trains a diffusion model — the same kind of model that generates images by removing noise step by step — to generate entire robot/agent trajectories (sequences of states and actions) instead of pictures. Because the model produces a whole trajectory at once rather than one step at a time, "sampling from the model" and "planning a good sequence of actions" become almost the same operation: you just nudge the denoising process with extra signals (a learned reward gradient, or fixed start/goal states) to steer it toward trajectories that are both realistic and useful. This single trick — called Diffuser — turns two well-known image-diffusion tricks, classifier guidance and inpainting, into reward-maximizing planning and goal/constraint-satisfying planning, respectively. Tested on long-horizon sparse-reward mazes, multi-task goal reaching, block-stacking, and standard offline-RL locomotion benchmarks, Diffuser matches or beats prior model-free and model-based baselines, especially when long horizons, sparse rewards, or new test-time goals are involved, while also being adaptable (reuse one trained model for many reward functions) and speedable (via a "warm-start" trick).

## Reading map

The paper's arc: (1) diagnose why the standard recipe — learn a dynamics model, then hand it to a classical trajectory optimizer — tends to fail (optimizers exploit the learned model's blind spots); (2) propose fusing the model and the planner into one object: a diffusion model over whole trajectories, so that "generate a sample" and "solve a planning problem" are nearly the same act; (3) show that two standard diffusion-model tricks have natural reinforcement-learning readings: classifier-guided sampling becomes reward-guided planning (Section 3.2), and inpainting becomes goal/constraint-conditioned planning (Section 3.3); (4) enumerate the useful properties this design buys for free — long-horizon robustness, task compositionality, temporal compositionality (stitching), variable-length plans (Section 4); (5) validate empirically on Maze2D (long-horizon, sparse reward, multi-task), block stacking (test-time flexibility/constraint composition), and D4RL locomotion (heterogeneous offline data), plus a runtime/warm-start study (Section 5); (6) situate against related generative-model-for-MBRL work (Section 6).

Key terms introduced in order: *trajectory optimization* and objective $J$ (Sec 2.1) → *diffusion probabilistic model* $p_\theta(\tau)$, forward/reverse process (Sec 2.2) → perturbed distribution $\tilde p(\tau) \propto p(\tau)h(\tau)$ (Eq. 1, Sec 3) → *temporal locality* / receptive field (Sec 3.1) → trajectory-as-2D-array representation (Eq. 2) → *control-as-inference* optimality variable $O_t$ and classifier-guidance analogy (Sec 3.2, Eq. 3, Algorithm 1) → *inpainting* as constraint satisfaction (Sec 3.3) → the four named "properties" (Sec 4) → warm-starting (Sec 5.4).

---

## Technical Walkthrough

### Abstract

**Role in the paper:** Compresses the whole contribution into one paragraph before any setup.

[P1] Model-based RL conventionally learns only an approximate dynamics model $f$ and hands the rest of decision-making to a classical trajectory optimizer. The paper instead proposes folding as much of the trajectory-optimization pipeline as possible into the generative model itself, so that sampling from the model and planning with it become nearly identical operations. The technical vehicle is a diffusion probabilistic model that plans by iteratively denoising trajectories. Classifier-guided sampling and image inpainting are reinterpreted as coherent planning strategies (reward-guided planning and goal/constraint-conditioned planning, respectively), and the resulting framework, Diffuser, is shown to be effective in control settings emphasizing long-horizon decision-making and test-time flexibility.

### 1 Introduction

**Role in the paper:** Motivates the problem, diagnoses the failure mode of the standard approach, and previews Diffuser's design principles and named properties.

[P2] Planning with a learned model is conceptually appealing because it isolates learning to what it does best — approximating unknown dynamics $f$ in a supervised-learning-like setting — and then reuses well-understood classical trajectory optimizers (e.g., direct collocation, iterative LQR-style methods) on top of that learned model.

[P3] In practice this combination rarely works as intended: powerful trajectory optimizers exploit the learned model, producing plans that resemble adversarial examples rather than genuinely optimal trajectories. As a consequence, most contemporary model-based RL algorithms borrow more from model-free RL (value functions, policy gradients) than from the trajectory-optimization toolbox, and those that do plan online restrict themselves to simple, gradient-free routines (random shooting, cross-entropy method) specifically to avoid triggering this exploitation failure.

[P4] The paper's alternative is to train a model that is directly amenable to trajectory optimization — i.e., design the model *for the planning problem it will be used in*, rather than as a faithful proxy for environment dynamics. Concretely: action distributions matter as much as state dynamics, long-horizon accuracy matters more than single-step error, and the model should stay agnostic to the reward function so a single trained model can serve multiple tasks in the same environment.

[P5] A further desideratum: the model's *plans*, not merely its one-step predictions, should improve with more data/training, and the resulting planner should avoid the myopic failure modes that plague standard shooting-based planning algorithms on long horizons.

[P6] These desiderata are instantiated as Diffuser, a trajectory-level diffusion probabilistic model (building on Sohl-Dickstein et al. 2015; Ho et al. 2020). Unlike standard model-based planners, which predict forward in time autoregressively, Diffuser predicts every timestep of a plan simultaneously. Diffusion models' iterative sampling process naturally supports flexible conditioning, letting auxiliary "guides" bias sampling toward high-return trajectories or trajectories satisfying constraints.

[P7] The paper enumerates four resulting properties: (i) **Long-horizon scalability** — trained on whole-trajectory accuracy rather than single-step error, so it avoids the compounding rollout error of single-step dynamics models; (ii) **Task compositionality** — reward gradients are just auxiliary signals added during sampling, so multiple reward terms can be composed by summing their gradients; (iii) **Temporal compositionality** — global coherence emerges from iteratively enforcing *local* consistency, letting the model generalize by stitching together in-distribution subsequences; (iv) **Effective non-greedy planning** — because the training objective (better trajectory predictions) and the planning objective (better plans) are the same objective, improving the model directly improves planning on long-horizon, sparse-reward tasks that defeat conventional planners.

[P8] The core contribution is summarized as a denoising diffusion model designed specifically for trajectory data plus an associated probabilistic framework for behavior synthesis — an unconventional but, per the authors, effective design for offline control problems requiring long-horizon reasoning and test-time flexibility.

### 2 Background

**Role in the paper:** Establishes notation and the two theoretical ingredients (trajectory optimization, diffusion models) the method will fuse.

[P9] *Problem setting.* A system evolves by discrete-time dynamics $s_{t+1} = f(s_t, a_t)$. Trajectory optimization seeks an action sequence $a_{0:T}$ maximizing an objective $J(s_0, a_{0:T}) = \sum_{t=0}^{T} r(s_t, a_t)$ over a planning horizon $T$. The shorthand $\tau = (s_0, a_0, s_1, a_1, \dots, s_T, a_T)$ denotes an interleaved state-action trajectory, and $J(\tau)$ its objective value.

[P10] *Diffusion probabilistic models* (Sohl-Dickstein et al. 2015; Ho et al. 2020) model data generation as an iterative denoising process $p(\tau^{i-1}\mid\tau^i)$, the reverse of a forward corruption process $q(\tau^i\mid\tau^{i-1})$ that gradually adds noise. The induced data distribution is $p_\theta(\tau^0) = \int p(\tau^N)\prod_{i=1}^N p_\theta(\tau^{i-1}\mid\tau^i)\, d\tau^{1:N}$, with $p(\tau^N)$ a standard Gaussian prior and $\tau^0$ the noiseless data. Parameters $\theta$ are fit by minimizing a variational bound on the negative log-likelihood, $\theta = \arg\min_\theta -\mathbb{E}[\log p_\theta(\tau^0)]$; the reverse process is typically parameterized as Gaussian with fixed, timestep-dependent covariance: $p_\theta(\tau^{i-1}\mid\tau^i) = \mathcal{N}(\tau^{i-1}\mid \mu_\theta(\tau^i,i), \Sigma^i)$. The forward process $q$ is usually prespecified rather than learned.

[P11] *Notation.* Superscripts $i$ index diffusion timestep; subscripts $t$ index planning timestep (e.g., $s_t^0$ is the $t$-th state of a noiseless trajectory). Superscripts on noiseless quantities ($\tau = \tau^0$) are dropped when unambiguous, and $s_t^\tau$/$a_t^\tau$ denote the $t$-th state/action within trajectory $\tau$.

### 3 Planning with Diffusion

**Role in the paper:** States the central design move — fusing model and planner — and formalizes it as sampling from a perturbed distribution.

[P12] Classical trajectory optimization needs access to dynamics $f$; the usual fix, learning an approximate $f$ and feeding it to a conventional planner, fails because learned models are poorly matched to planners designed for ground-truth models — the planner ends up finding adversarial-example-like trajectories rather than good ones.

[P13] The proposed fix is tighter coupling: instead of a learned model plugged into an external planner, subsume as much of planning as possible into the generative model itself, so planning becomes nearly identical to sampling from a diffusion model of trajectories $p_\theta(\tau)$. Flexible conditioning is achieved by sampling instead from a perturbed distribution $\tilde p(\tau) \propto p(\tau)\,h(\tau)$ (Eq. 1), where $h(\tau)$ encodes prior evidence (e.g., an observation history), desired outcomes (a goal), or objectives to optimize (rewards/costs). Sampling from $\tilde p$ is a probabilistic analogue of the Section 2.1 optimization problem: it must be both *physically realistic* under $p(\tau)$ and *high-reward/constraint-satisfying* under $h(\tau)$. Because dynamics information ($p(\tau)$) is separated from the task-specific perturbation ($h(\tau)$), one trained diffusion model can be reused across multiple tasks in the same environment.

#### 3.1 A Generative Model for Trajectory Planning

**Role in the paper:** Works out the concrete architectural constraints implied by the fused model/planner design.

[P14] *Temporal ordering.* Since sampling and planning are now the same act, autoregressive (forward-in-time) prediction is no longer viable: goal-conditioned inference like $p(s_1\mid s_0, s_T)$ requires $s_1$ to depend on a *future* state $s_T$ as well as a past one $s_0$. This reflects a general asymmetry — dynamics prediction is causal (present determined by past), but decision-making is anti-causal (present decisions conditioned on desired futures). Diffuser therefore predicts all timesteps of a plan concurrently rather than autoregressively.

[P15] *Temporal locality.* Diffuser is neither autoregressive nor Markovian, but it retains a relaxed local structure: a single denoising step, implemented with a temporal convolution, has a receptive field limited to nearby (past and future) timesteps, so it can only enforce *local* consistency at each step. Stacking many such local-consistency-enforcing steps together nonetheless drives *global* coherence of the sampled plan.

[P16] *Trajectory representation.* Because the controller's plan quality matters as much as raw state-prediction quality, states and actions are predicted jointly, with actions treated simply as extra state dimensions. Trajectories are represented as a 2D array with one column per planning timestep: $\tau = \begin{pmatrix} s_0 & s_1 & \cdots & s_T \\ a_0 & a_1 & \cdots & a_T \end{pmatrix}$ (Eq. 2).

[P17] *Architecture.* Three requirements follow: (1) non-autoregressive whole-trajectory prediction, (2) temporally-local denoising steps, (3) equivariance along the horizon axis but not along the state/action-feature axis. These are satisfied by a model of repeated temporal convolutional residual blocks resembling image-diffusion U-Nets, but with 1D temporal convolutions replacing 2D spatial ones (architecture diagram in Figure A1). Because the model is fully convolutional, the planning horizon is set by input dimensionality, not architecture, and can be changed dynamically at planning time.

[P18] *Training.* Diffuser parameterizes a learned noise/gradient estimate $\epsilon_\theta(\tau^i, i)$ of the denoising process, from which the reverse-process mean $\mu$ can be solved in closed form (Ho et al. 2020). It is trained with the simplified objective $\mathcal{L}(\theta) = \mathbb{E}_{i,\epsilon,\tau^0}\left[\lVert \epsilon - \epsilon_\theta(\tau^i, i)\rVert^2\right]$, where $i \sim \mathcal{U}\{1,\dots,N\}$ is the diffusion timestep, $\epsilon \sim \mathcal{N}(0,I)$ is the noise target, and $\tau^i$ is $\tau^0$ corrupted by $\epsilon$. Reverse-process covariances follow the cosine noise schedule of Nichol & Dhariwal (2021).

#### 3.2 Reinforcement Learning as Guided Sampling

**Role in the paper:** Derives reward-maximizing planning as classifier-guided diffusion sampling.

[P19] Using the control-as-inference framework (Levine, 2018), let $O_t$ be a binary "optimality" variable for timestep $t$, with $p(O_t=1) = \exp(r(s_t,a_t))$. Setting $h(\tau) = p(O_{1:T}\mid\tau)$ in Eq. 1 gives $\tilde p(\tau) = p(\tau\mid O_{1:T}=1) \propto p(\tau)\,p(O_{1:T}=1\mid\tau)$ — i.e., the RL problem is recast as conditional sampling from the trajectory diffusion model.

[P20] This is exactly the setup used for classifier-guided image generation (Dhariwal & Nichol, 2021): when $p(O_{1:T}\mid\tau^i)$ is sufficiently smooth, the reverse diffusion transitions can be approximated as Gaussian, $p(\tau^{i-1}\mid\tau^i, O_{1:T}) \approx \mathcal{N}(\tau^{i-1}; \mu+\Sigma g, \Sigma)$ (Eq. 3), where $\mu,\Sigma$ are the unguided reverse-process parameters and the guidance term is $g = \nabla \log p(O_{1:T}\mid\tau)\big|_{\tau=\mu} = \sum_{t=0}^T \nabla_{s_t,a_t} r(s_t,a_t)\big|_{(s_t,a_t)=\mu_t} = \nabla J(\mu)$ — the gradient of the return with respect to the (mean) trajectory.

[P21] Practically: train a diffusion model $p_\theta(\tau)$ on all available trajectory data; separately train a return predictor $J_\phi$ that predicts cumulative reward of noisy trajectory samples $\tau^i$; use $\nabla J_\phi$ to shift the reverse-process means per Eq. 3 during sampling. The first action of the resulting sampled trajectory $\tau \sim p(\tau\mid O_{1:T}=1)$ is executed in the environment, and planning repeats in a standard receding-horizon loop (pseudocode: Algorithm 1).

#### 3.3 Goal-Conditioned RL as Inpainting

**Role in the paper:** Derives constraint-satisfying planning as diffusion-model inpainting.

[P22] Some planning problems are better framed as constraint satisfaction than reward maximization — e.g., "reach a goal location," where any feasible trajectory meeting the constraint is acceptable. Under the 2D trajectory-array representation (Eq. 2), this becomes an inpainting problem: state/action constraints play the role of observed pixels, and the diffusion model fills in ("inpaints") all unconstrained entries consistently with them.

[P23] The required perturbation function is a Dirac delta at observed/constrained values and constant elsewhere: for a state constraint $c_t$ at timestep $t$, $h(\tau) = \delta_{c_t}(s_0,a_0,\dots,s_T,a_T) = \delta$ if $c_t = s_t$, else a constant (action constraints are defined identically). In practice this is implemented by sampling from the *unperturbed* reverse process $\tau^{i-1}\sim p(\tau^{i-1}\mid\tau^i)$ and then overwriting the sampled values with the conditioning values at every diffusion timestep $i \in \{0,\dots,N\}$.

[P24] Even pure reward-maximization runs require this inpainting-style conditioning, because every sampled trajectory must begin at the agent's actual current state — this is the "constrain first state of plan" step (line 10 of Algorithm 1).

### 4 Properties of Diffusion Planners

**Role in the paper:** Argues, with qualitative demonstrations (Figure 3), why the fused model/planner design is advantageous beyond raw benchmark numbers.

[P25] *Learned long-horizon planning.* Standard single-step dynamics models are agnostic to the planning algorithm that will use them; Diffuser's planning routine (Algorithm 1), in contrast, is tightly coupled to diffusion sampling itself — planning differs from sampling only by the guidance term $h(\tau)$. Consequently, Diffuser's effectiveness as a long-horizon *predictor* directly transfers to effectiveness as a long-horizon *planner*, letting it succeed on sparse-reward goal-reaching tasks that defeat shooting-based methods (Figure 3a).

[P26] *Temporal compositionality.* Because global coherence arises from iteratively repairing local consistency (Section 3.1) rather than from strict Markov transitions, Diffuser can stitch together familiar trajectory subsequences into novel, out-of-distribution trajectories. Demonstration: trained only on straight-line trajectories, it generalizes to V-shaped trajectories by composing two straight segments at their intersection point (Figure 3b).

[P27] *Variable-length plans.* Since the model is fully convolutional along the horizon dimension, the planning horizon is not fixed by architecture — it is set by the size of the initializing noise sample $\tau^N \sim \mathcal{N}(0,I)$, enabling variable-length plans at inference time (Figure 3c).

[P28] *Task compositionality.* Diffuser encodes environment dynamics/behavior but stays independent of any specific reward function; it acts as a prior over plausible futures that can be steered by comparatively lightweight perturbation functions $h(\tau)$, including sums of multiple perturbations. This is demonstrated by planning successfully for a reward function never seen during diffusion-model training (Figure 3d).

### 5 Experimental Evaluation

**Role in the paper:** Tests three claimed capabilities — long-horizon planning without reward shaping, generalization to new goal configurations, and recovering good controllers from heterogeneous offline data — plus a runtime study.

[P29] The evaluation targets (1) long-horizon planning without manual reward shaping, (2) generalization to goal configurations unseen during training, and (3) recovering an effective controller from heterogeneous, mixed-quality offline data, followed by a study of practical runtime/speed trade-offs.

#### 5.1 Long Horizon Multi-Task Planning

[P30] Maze2D environments (Fu et al., 2020) require traversing to a goal that gives reward 1, with no reward anywhere else; reaching the goal can take hundreds of steps, which defeats credit assignment for even strong model-free baselines (Table 1).

[P31] Diffuser plans via the inpainting strategy, conditioning on start and goal location (the goal is identifiable to model-free baselines too, as the only nonzero-reward state in the data) and executing the sampled trajectory open-loop. Diffuser scores over 100 in all three maze sizes — i.e., it outperforms a reference expert policy — visualized as an evolving denoising process in Figure 4.

[P32] In the multi-task variant (Multi2D, goal resampled every episode), Diffuser needs no retraining — only the conditioning goal changes — and performs as well multi-task as single-task. The strongest model-free baseline (IQL) drops substantially when adapted to multi-task via hindsight relabeling (Appendix A). MPPI, despite using ground-truth dynamics, performs poorly relative to Diffuser, underscoring that long-horizon planning is hard even without model-inaccuracy issues.

#### 5.2 Test-time Flexibility

[P33] A block-stacking suite tests three settings: Unconditional Stacking (build the tallest tower), Conditional Stacking (build a tower in a specified block order), and Rearrangement (match a reference arrangement of blocks). All methods train on 10,000 PDDLStream-generated demonstration trajectories (Garrett et al., 2020), with reward 1 for successful placements.

[P34] A single trained Diffuser is reused across all three settings, only swapping the perturbation function $h(\tau)$: unconditional sampling from $p(\tau)$ for Unconditional Stacking; for Conditional Stacking / Rearrangement, two composed perturbations — one maximizing likelihood that the final state matches the goal configuration, one enforcing an end-effector/cube contact constraint (details in Appendix B).

[P35] Diffuser substantially outperforms BCQ and CQL baselines (Table 3), with the conditional settings — which demand flexible, on-the-fly behavior generation into states not seen in training — proving especially hard for the model-free baselines (visual example, Figure 5).

#### 5.3 Offline Reinforcement Learning

[P36] On the D4RL offline locomotion suite (Fu et al., 2020), Diffuser combines the Section 3.2 reward-guided sampling (guiding toward high-reward regions using a learned $J_\phi$ trained on the same trajectory data) with the Section 3.3 inpainting conditioning on the current state. Baselines span model-free RL (CQL, IQL), return-conditioning (Decision Transformer), and model-based RL (Trajectory Transformer, MOPO, MOReL, MBOP).

[P37] Diffuser performs comparably to prior methods overall (Table 2) — better than MOReL, MBOP, and DT, though worse than the best single-task-specialized offline methods. A control experiment using Diffuser purely as a dynamics model inside a conventional trajectory optimizer (MPPI) performed no better than random, indicating that Diffuser's effectiveness comes specifically from the coupled model+planner design, not from superior open-loop predictive accuracy alone.

#### 5.4 Warm-Starting Diffusion for Faster Planning

[P38] A practical limitation is that individual plans are slow to generate due to the iterative denoising process, and naive open-loop execution would require regenerating a full plan at every environment step. The proposed fix: warm-start each new plan by running a limited number of forward-noising steps on the *previous* plan, then denoising only that partially-noised trajectory for a matching number of reverse steps. Varying the number of denoising steps from 2 to 100 (Figure 7) shows the planning budget can be cut markedly with only modest performance loss, provided each new plan is warm-started from the prior one.

### 6 Related Work

**Role in the paper:** Positions Diffuser against prior generative-model-based MBRL work and prior diffusion-model applications.

[P39] Prior deep generative modeling for MBRL (convolutional U-nets, stochastic recurrent networks, latent world models, vector-quantized autoencoders, neural ODEs, normalizing flows, GANs, EBMs, GNNs, NeRFs, Transformers) generally preserves an abstraction barrier: learning approximates dynamics, and a separate, model-agnostic planning or policy-optimization algorithm is applied afterward. Related efforts to break this barrier (latent-space reward prediction, value-weighted model training, collocation on learned energies) still largely keep model and planner as separate components trained with different objectives. Diffuser instead trains model and planner jointly by generating all trajectory timesteps concurrently and conditioning with auxiliary guidance functions. Separately, diffusion models are established for images, waveforms, 3D shapes, and text, and are connected theoretically to score matching and energy-based models, but — to the authors' knowledge — had not previously been applied to reinforcement learning or decision-making before this work.

### 7 Conclusion

**Role in the paper:** Restates the contribution and its practical payoffs.

[P40] Diffuser is a denoising diffusion model for trajectory data; planning with it is almost identical to sampling from it, differing only by the addition of perturbation functions that guide the sample. The resulting learned planning procedure handles sparse rewards gracefully, plans for new reward functions without retraining, and — via its temporal compositionality — can produce out-of-distribution trajectories by stitching in-distribution subsequences, pointing toward a new class of diffusion-based planning procedures for model-based RL.

### Appendix highlights (condensed)

**Role in the paper:** Supplementary implementation and baseline details; included briefly because they clarify how the method was actually run (full appendix baseline bookkeeping and citation sourcing are omitted as boilerplate — see Coverage notes).

[P41] *Architecture and training hyperparameters (Appendix C).* Diffuser uses a U-Net with 6 repeated residual blocks (two temporal convolutions + group norm + Mish nonlinearity per block; diffusion-timestep embeddings injected via a fully-connected layer into the first convolution of each block), trained with Adam (lr $4\times10^{-5}$, batch size 32) for 500k steps. The return predictor $J_\phi$ reuses the first half of the U-Net plus a final linear layer. Planning horizon $T$ = 32 (locomotion), 128 (block-stacking and Maze2D/Multi2D U-Maze), 265 (Medium maze), 384 (Large maze). Diffusion steps $N$ = 20 (locomotion), 100 (block-stacking). Guide scale $\alpha$ = 0.1 for most tasks (0.0001 for hopper-medium-expert); return discount $\gamma=0.997$ (insensitive above 0.99). Predicting noise $\epsilon$ vs. clean data $\tau^0$ made little difference to control performance; horizon could be shortened if the guide scale was correspondingly reduced.

[P42] *Block-stacking perturbation functions (Appendix B).* "Final state matching" is a learned per-timestep classifier $h_{\text{match}}$ trained on demonstration data to detect whether a state shows block A stacked on block B. The "contact constraint" $h_{\text{contact}}(\tau) = \sum_{i=0}^{64} \alpha^{-1}\lVert c_i - 1\rVert^2$ penalizes deviation from Kuka-arm/block-A contact over the first 64 timesteps of a trajectory (i.e., during initial grasping contact).

[P43] *Baseline tuning notes (Appendix A).* IQL (single- and multi-task, the latter via hindsight-relabeling-style goal sampling from a geometric distribution over future states) was tuned over temperature and expectile; CQL and BCQ (block-stacking) were tuned over discount factor and $\tau$; most other baseline numbers are taken directly from their original papers' tables, with sources cited per task.

---

## Plain-Language Walkthrough

### Abstract

[P1] Normally, when a robot uses "learned models" to plan, it learns a rough simulator of the world and then hands that simulator to a separate, off-the-shelf planning algorithm. This paper does something different: it trains one model that generates a *whole imagined plan at once*, the same way an image-generating AI paints a whole picture by starting from noise and gradually cleaning it up. Because the model *is* the planner, two well-known tricks from image generation — steering an image toward a class label, and filling in missing patches of a photo — become, respectively, "steer the plan toward high reward" and "fill in a plan given a fixed start and end point."

### 1 Introduction

[P2] The old recipe sounds sensible: learn a model of "if I do X, what happens next," then let a classical, well-understood planning algorithm figure out the best sequence of actions using that model.

[P3] The problem is that clever planning algorithms are very good at finding the *cracks* in an imperfect learned model — they discover action sequences that trick the model into predicting great outcomes, even though those sequences would fail in the real world (a bit like finding a glitch in a video game rather than actually beating the level fairly). Because of this, most practical systems either avoid heavy-duty planning algorithms altogether, or only use very simple, cautious ones.

[P4] The paper's fix: build a model whose whole reason for existing is to be planned with — not a general-purpose simulator that happens to get planned over afterward. Concretely, the model should care about getting whole action sequences right, not just the very next step, and it shouldn't be tied to any one specific goal or reward, so one trained model can be reused for many different tasks in the same world.

[P5] Also, more experience should make the *plans* better, not just the model's raw predictions — and the system should avoid the classic "shortsighted" mistake where a planner grabs a small reward now and misses a bigger one down the road.

[P6] The authors build this as "Diffuser," which borrows the "denoising" trick from image-generating AI (the kind that starts from random static and gradually sharpens it into a photo) but applies it to whole trajectories — sequences of states and actions — generating the entire sequence together rather than one step at a time.

[P7] Four payoffs of this design: it handles long tasks well because it was trained to get the whole trajectory right, not just one step; it can combine multiple goals/rewards just by adding their "nudges" together during generation; it can string together familiar pieces of past experience into new, never-seen routes; and because "getting better at generating trajectories" and "getting better at planning" are literally the same training signal, it naturally avoids the shortsightedness that trips up simpler planners.

[P8] In short: a purpose-built denoising model for trajectories, plus a recipe for using it to make decisions, aimed especially at long, sparse-reward tasks and situations where you want flexibility about what you're trying to achieve.

### 2 Background

[P9] Setup: the world evolves by "next state = some function of current state and action." Planning means picking a sequence of actions that adds up to the best total reward over some time horizon. A "trajectory" is just shorthand for a full recorded sequence of states and actions.

[P10] A diffusion model is an AI that learns to generate realistic data (originally images) by learning to reverse a "static-adding" process: start with a clean example, gradually blur it into pure noise, and train the model to undo that blurring, one small step at a time. Run backward from pure noise, that undoing process manufactures a brand-new, realistic-looking example.

[P11] Bookkeeping note: there are two different notions of "step" in this paper — the model's internal denoising steps (going from noisy to clean) and the timesteps of the actual plan (step 1, step 2, ... of the robot's trajectory). The paper uses superscripts for the former and subscripts for the latter.

### 3 Planning with Diffusion

[P12] Classic planning needs a decent simulator of the world, and using a learned (imperfect) one invites the planner to exploit its flaws rather than genuinely solve the task.

[P13] The fix: don't separate "simulator" from "planner" — build a single diffusion model of entire trajectories, and get plans by "sampling" from it while nudging the sampling process toward what you want (call that nudge $h$). A generated trajectory needs to satisfy two things at once: it has to look like something the world could actually do, and it has to score well on whatever you're nudging it toward (a goal, a reward, a rule). Because the "realistic world behavior" part and the "what I want" part are kept separate, you can reuse the same trained model for lots of different goals in the same environment.

#### 3.1 A Generative Model for Trajectory Planning

[P14] Normally you'd predict a trajectory one timestep after another, like predicting the next word in a sentence. But planning often needs to reason about a *future* goal influencing an *earlier* action ("go left now because I need to end up at the exit later"), which breaks a simple forward-only prediction order. So Diffuser generates every timestep of the plan all at the same time, rather than one after another.

[P15] Even though it's not doing strict step-by-step causal prediction, each single "cleanup" pass only looks at nearby timesteps — like a blurry photo being sharpened locally, patch by patch. But by repeating this local sharpening many times, the whole trajectory ends up globally sensible, the same way repeatedly touching up nearby brushstrokes in a painting eventually produces a coherent whole picture.

[P16] A plan is stored as a simple grid: one column per timestep, with the state and the action stacked in that column. Actions are just treated as extra numbers alongside the state, predicted together.

[P17] The network design mirrors the "U-Net" architecture used in image-generating AI, just swapped from 2D image patches to 1D sequences-over-time. One nice side effect: because it's built from convolutions applied uniformly along time, the length of the plan isn't hard-wired into the network — you can ask for a longer or shorter plan just by changing how much noise you start from.

[P18] Training just means: take a real trajectory, add a random amount of noise to it, and train the network to guess what noise was added (so it can later subtract it back out). This is the exact same training trick used in image diffusion models, just applied to trajectories instead of pictures.

#### 3.2 Reinforcement Learning as Guided Sampling

[P19] Think of "getting reward" as a coin flip that's more likely to land heads the more reward you're earning at each moment. Wanting a *good* trajectory is then the same as asking: "generate a trajectory *conditional on* all those coin flips having landed heads."

[P20] This is mathematically the same trick used to make image-generating AIs produce, say, "a picture of a cat" instead of a random picture: nudge the noise-removal process using the gradient of a separately trained "how good is this?" scorer. In this paper, that scorer estimates expected reward, and its gradient tells you which direction to nudge the in-progress (partially denoised) trajectory to make it more rewarding.

[P21] In practice: train the trajectory-generating model on all your data; separately train a small "reward guesser" network; while generating a plan, keep nudging it, at every denoising step, in the direction that guesser says increases reward. Take just the first action of the finished plan, do it in the real world, observe what happens, and re-plan from scratch — repeating this loop as you go (this is standard "replan every step" control).

#### 3.3 Goal-Conditioned RL as Inpainting

[P22] Some tasks aren't about maximizing a score — they're about satisfying a requirement, like "end up at this particular spot," and any trajectory that does so is fine. Using the grid representation of a plan, this becomes exactly like photo inpainting: you have some cells of the grid pinned down (start state, goal state) and you ask the model to fill in the rest so the whole picture makes sense.

[P23] Concretely, wherever you've fixed a value (say, "state at time 0 must be here"), you clamp that cell to the fixed value; everywhere else is left free for the model to fill in. In practice this is done by letting the model do its normal denoising, but after every single denoising step, forcibly resetting the pinned cells back to their required values — like repeatedly re-drawing a photo's border while letting the AI keep touching up the middle.

[P24] Even when you're maximizing reward (Section 3.2), you still need this "pin the start state" trick, since every plan has to actually start from wherever the robot currently is.

### 4 Properties of Diffusion Planners

[P25] Because the "planner" and the "generator" are literally the same machinery, a model that's simply *good at imagining realistic long trajectories* automatically becomes a *good long-horizon planner* — it doesn't get lost or shortsighted the way simpler goal-reaching methods do on tasks with a long delay before any reward shows up.

[P26] The model can produce genuinely novel routes it never saw exactly during training, by gluing together pieces of routes it *did* see — demonstrated by training only on straight-line paths and then generating a V-shaped path by combining two straight segments at their meeting point. This is a bit like an artist who's only ever painted straight roads being able to paint a fork in the road by combining what they know about two straight roads.

[P27] Since the model isn't hard-wired to a fixed plan length, you can ask for shorter or longer plans just by changing how much starting noise you feed it — no retraining needed.

[P28] The model itself doesn't "know" about any specific goal or reward — it just knows what realistic behavior looks like. So you can bolt on a brand-new goal or reward it's never seen before, purely by changing the nudging signal at planning time, and it will still generate sensible plans toward that new goal.

### 5 Experimental Evaluation

[P29] The experiments test three things: can it plan over very long tasks without hand-engineered reward shaping; can it adapt to new goals it wasn't specifically trained for; and can it learn a good controller from a mixed bag of so-so training data. There's also a practical speed test at the end.

#### 5.1 Long Horizon Multi-Task Planning

[P30] In the maze tasks, the only reward is a single point far away — everywhere else is reward-zero — so it can take hundreds of steps of "doing nothing rewarding" before reaching the payoff. This kind of long, silent buildup is exactly what trips up typical learning algorithms, because they struggle to figure out which early actions deserve credit for a reward that arrives much later.

[P31] Diffuser is simply told "start here, end there" (via the inpainting trick) and fills in a full route through the maze in one shot, which is then followed step by step. It beats even a hand-crafted expert route through the maze.

[P32] When the goal is randomized every episode instead of fixed, Diffuser doesn't need to be retrained at all — you just tell it a different goal each time, and it performs just as well. The best competing method loses a lot of performance when forced to handle random goals this way. Interestingly, even a classical planner that's handed the *true* (not learned) simulator of the maze still does much worse than Diffuser, showing that the long-horizon-planning problem itself is hard, not just the "the model might be wrong" problem.

#### 5.2 Test-time Flexibility

[P33] Three block-stacking challenges: stack blocks as high as possible with no other rules; stack them in a specific order; and rearrange blocks to match a reference layout. All the training data comes from a symbolic block-stacking planner's demonstrations.

[P34] One single trained Diffuser handles all three challenges — the only thing that changes between them is what "nudge" is applied during generation. For the harder challenges, two nudges are combined at once: one that pushes the final stack toward the right block order, and one that enforces physically sensible contact between the robot arm and the block while grabbing it.

[P35] Diffuser clearly outperforms two standard model-free comparison methods, especially on the trickier goal-specific stacking tasks, where those methods struggle to improvise appropriately.

#### 5.3 Offline Reinforcement Learning

[P36] On a standard set of "learn a good walking/running controller from a fixed pile of past data" benchmarks, Diffuser combines both tricks at once: nudge toward high reward (Section 3.2) and pin the plan to start from wherever the robot currently is (Section 3.3).

[P37] Diffuser holds its own against a wide range of prior methods — better than some, a bit behind the very best specialized methods. A telling side experiment: if you instead just use Diffuser as a plain simulator plugged into an old-school planning algorithm (rather than letting it plan directly), performance collapses to random. This confirms that Diffuser's advantage really comes from fusing the model and the planner together, not just from the model being an accurate simulator.

#### 5.4 Warm-Starting Diffusion for Faster Planning

[P38] The catch with this whole approach is that generating one plan takes many denoising steps, which is slow, and naively you'd have to regenerate a full plan from scratch at every single moment of execution. The fix is to reuse most of the *previous* plan: only lightly re-noise it and then quickly re-clean it up, instead of starting over from pure noise. This lets the system cut its "thinking time" by roughly a factor of ten with only a small dip in how well it performs.

### 6 Related Work

[P39] Lots of prior work has tried using fancy AI models (various neural network styles) as the "simulator" component of model-based RL, but almost all of them keep the simulator and the planning algorithm as two separate pieces bolted together afterward. Diffusion models themselves are well established for generating images, audio, 3D shapes, and text, but — as far as the authors know — nobody had used them for robot decision-making before this paper; this work fuses the simulator and the planner into one trained object instead of keeping them separate.

### 7 Conclusion

[P40] The headline idea: build one model where "generate a plausible trajectory" and "come up with a good plan" are basically the same action, differing only by a small nudge. This buys graceful handling of tasks with rare/delayed rewards, easy adaptation to brand-new goals without retraining, and the ability to invent new routes by recombining pieces of familiar ones.

### Appendix highlights (condensed)

[P41] Nuts-and-bolts settings: the network is a fairly standard "U-Net"-style stack of 6 blocks trained for 500,000 steps; plan lengths and the number of denoising steps used differ by task (longer mazes get longer plans and more denoising steps); a "guide strength" dial controls how hard the reward nudge pushes, and it needed careful tuning per task; whether the network predicted "the noise" or "the clean answer directly" barely mattered for final performance.

[P42] For block stacking, the two nudges mentioned earlier are: a learned "does this look like block A on block B?" detector, and a physics-style penalty that punishes the plan if the robot arm isn't properly touching the block during the grab.

[P43] The comparison methods (other algorithms Diffuser is benchmarked against) were tuned with modest hyperparameter sweeps and, where possible, their numbers were taken directly from the original papers that introduced them, to keep comparisons fair.

---

## Coverage notes

- **Excluded as boilerplate:** Author affiliations/correspondence footnotes, "Code References" (library citations), "Acknowledgements," and the full "References" list (~70 entries) were omitted — pure bibliographic/administrative content with no substantive claims.
- **Condensed:** Section 6 (Related Work) was compressed into a single paragraph pair (P39) per walkthrough rather than exhaustively cataloguing every cited modeling paradigm (U-Nets, stochastic recurrent nets, VQ-autoencoders, neural ODEs, normalizing flows, GANs, EBMs, GNNs, NeRFs, Transformers, etc.), since these function mainly as a citation survey rather than argument-bearing content; the shared conclusion (prior work keeps model and planner separate; this paper doesn't) is preserved.
- **Appendices A–C** (baseline hyperparameter tuning ranges, scoring-table sourcing citations, and full architectural diagrams) are supplementary and were folded into three condensed paragraphs (P41–P43) rather than given full paragraph-by-paragraph treatment, since most of their content is either (a) already summarized qualitatively in the main-text experimental sections, or (b) pure bookkeeping (which GitHub repo a baseline came from, which table in which other paper a number was copied from).
- **Tables 1–3** and Figures 1–7 are referenced inline at the relevant paragraphs (e.g., P31/Table 1, P35/Table 3, P37/Table 2, P38/Figure 7) rather than reproduced, since the surrounding prose already states the qualitative and, where relevant, headline quantitative takeaway (e.g., "Diffuser scores over 100 in all three maze sizes").
- **Equations** (1)-(3) and the trajectory-array notation (Eq. 2) are preserved in the Technical Walkthrough with symbols glossed in place; the Plain-Language Walkthrough replaces them entirely with the coin-flip/nudge/inpainting analogies and flags no additional nuance loss beyond what's already noted in P20/P23.
- No paragraph count exceeded the ~40 guardrail for the main text (40 substantive paragraphs, P1–P40); the three appendix paragraphs (P41–P43) are additional supplementary content beyond that count, included because the paper is short and the implementation details add concrete value without meaningfully lengthening the report.

## Reference

Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine. "Planning with Diffusion for Flexible Behavior Synthesis." *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162, 2022. arXiv:2205.09991v2 [cs.LG]. https://arxiv.org/abs/2205.09991 · Project page/code: https://diffusion-planning.github.io
