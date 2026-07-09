# Diffusion Models for Robotic Manipulation: A Survey

## TL;DR

This is a survey — the authors claim it's the first one focused specifically on diffusion models (DMs) in robotic manipulation — covering three application areas: trajectory generation, grasp synthesis, and visual data augmentation. It works through the two foundational diffusion formulations (score-based/NCSN and DDPM), catalogs the three dominant denoising-network architectures (CNN/temporal U-Net, transformer, MLP) and how each handles conditioning, surveys sampling-acceleration techniques (DDIM, DPM-solver, distillation, flow matching), and closes with a taxonomy of benchmarks plus two headline open problems: generalizability and slow inference. Diffusion Policy (Chi et al., 2023 — the paper covered in the companion report to this one) functions as the field's reference point throughout: it's the single most common baseline other methods compare against in the survey's own benchmark tables (Table 6).

## Architecture

There's no single architecture here — the paper's job is to catalog the field's choices, not propose one. Three things are worth pulling out:

**Two mathematical frameworks, both traced to Sohl-Dickstein et al. (2015):**
- **Score-based DMs / SMLD (Song and Ermon, 2019):** a Noise Conditional Score Network (NCSN) is trained to estimate ∇ₓ log p(xₖ|x) — the score — at each of *K* noise scales, using a weighted denoising score-matching loss (Eq. 1). Sampling recursively runs Langevin dynamics at each noise scale, annealing from high to low noise. The survey notes this original formulation is rarely used directly in robotic manipulation (it cites inefficient sampling as the likely reason) but treats it as necessary background because it underlies later SDE-based methods.
- **DDPM (Ho et al., 2020):** instead of the score, a network ε_θ is trained to directly predict the noise added at each step (Eq. 5), via a Markovian forward process (Eq. 3) that also has a closed-form single-step version (Eq. 4). One denoising step is taken per noise level rather than recursive Langevin dynamics per level. The survey states plainly that DDPM itself is also rarely used as-is in robotics — what's actually widely used is DDIM (Song et al., 2021a), which keeps DDPM's training procedure but changes only the sampling process, which is why the survey still walks through DDPM's derivation in full (Eqs. 3–7) before addressing its shortcomings.

**Three denoising-network architecture families (Section 3.1), with the survey's own comparative judgment (Section 3.1.4):**
- **CNN / temporal U-Net** — the most frequently used; originates from Janner et al.'s Diffuser (1D temporal convolutions replacing the 2D spatial convolutions of image U-Nets), adapted by Diffusion Policy to denoise action-only trajectories (not joint state-action) using [FiLM](#film-perez-et-al-2018) conditioning. Verdict: lower hyperparameter sensitivity than transformers, slightly better on some complex visual tasks with position control, but prone to over-smoothing high-frequency/velocity-control signals.
- **Transformer** — history of observations, diffusion timestep, and the noisy action are input tokens; conditioning via self- or cross-attention. Some methods use plain cross-attention transformers (as in Diffusion Policy), others adopt [Diffusion Transformers](#dit-peebles-and-xie-2023) (DiT, Peebles and Xie 2023) specifically. Verdict: best for high-dimensional inputs/outputs and long-range dependencies, at the cost of heavier compute and longer inference.
- **MLP** — mostly seen in RL contexts, taking concatenated observation/action/timestep features. Verdict: weakest with high-dimensional visual input, but cheapest and most training-stable, which is presumably why it's the RL-community default.
- Two conditioning mechanisms dominate: FiLM (used by CNN backbones, extensible to multiple concatenated conditions) and inpainting (Janner et al.'s original approach — replacing specific generated states with condition states at each denoising step; only supports point-wise conditions, so it's largely been superseded by FiLM for anything beyond simple goal-state conditioning).

**Robotics-specific adaptations the field had to add on top of vanilla DMs (Section 2.3):** (1) conditioning on observations at all — vanilla DMs generate unconditionally from noise, robots need the output tied to sensed state; (2) handling temporal correlation in trajectories, resolved almost universally via receding-horizon control — generate a subtrajectory of length *H*, execute only *H_c ≤ H* of it, then replan on updated observations (Fig. 2) — versus grasp poses, which are single actions generated in one backward pass from a single observation with no re-planning during execution.

Grasp-synthesis methods layer their own domain-specific architectural borrowings on top of these general choices — e.g., Song et al. (2024b) conditions 6-DoF grasp poses on grasp location and volumetric features following the [GIGA framework](#giga-framework-jiang-et-al-2021) (Jiang et al., 2021), applying a diffusion model on top of an implicit occupancy-and-affordance representation rather than GIGA's original direct regression.

## Data

The survey doesn't use a single dataset — it catalogs benchmarks used across the field (Section 5, Tables 6, 7, 9):
- **Trajectory/imitation-learning benchmarks:** CALVIN, RLBench, RelayKitchen/FrankaKitchen, Meta-World, D4RL Kitchen, FurnitureBench (real-world), Adroit (dexterous manipulation), LIBERO (lifelong learning), LapGym (medical/surgical).
- **Grasp-specific datasets:** Acronym, DA2, DexGraspNet, MultiDex, OakInk, VGN.
- **Demonstration counts vary by roughly five orders of magnitude across surveyed methods** — from ~20 demos (Vosylius et al., 2024) up to 6.54 million transitions (Saha et al., 2024) or 400,000 video clips (Bharadhwaj et al., 2024b). Table 6 makes this variance explicit but does not normalize for it.
- Most methods condition on RGB images; a growing subset condition on point clouds or point-cloud feature embeddings (for better geometric/occlusion handling at the cost of more hardware); a smaller subset use ground-truth state directly (harder to transfer sim-to-real, easier to iterate on).

## Training setup

Not a single recipe — the survey reports field-level tendencies rather than one training regime (Section 3.2): 50–100 training noise levels is the common range, with only 5–10 sampling steps at inference via DDIM; a few outliers go as low as 3–4 (Vosylius et al., 2024; Reuss et al., 2023) or as high as 20–30 (Mishra and Chen, 2024; Wang et al., 2024b) inference steps. Beyond this noise-schedule-and-step-count observation, the survey does not attempt to summarize optimizers, batch sizes, or learning-rate schedules across methods — those vary too much per architecture and paper to tabulate, and the survey doesn't claim to.

## Evaluation & results

**How methods get evaluated (Section 5):** many surveyed methods are only compared against non-diffusion baselines — the survey notes this explicitly rather than glossing over it. Where diffusion-based baselines are used, three recur most often: SE(3)-Diffusion Policy (Urain et al., 2023) for SE(3)/grasp-space work, {Diffuser, Diffusion-QL, Decision Diffuser} for offline-RL work, and **Diffusion Policy (Chi et al., 2023) as the single most common general-purpose baseline** — many later methods are explicitly built on top of it.

**Concrete cross-method numbers the survey reports (these are the original papers' numbers, restated, not independently re-run by this survey's authors):**
- 3D Diffusion Policy outperforms (2D) Diffusion Policy by an average of 24.2% success rate across Adroit, MetaWorld, and DexArt (74.4% vs. ~50%), and by 50% on 4 real-world tasks (85.0% vs. ~35%).
- 3D Diffuser Actor "greatly outperforms" 3D Diffusion Policy on CALVIN's zero-shot long-horizon tasks specifically — no real-world comparison is provided for this pair.
- On sampling speed: Ko et al. (2024) is cited as reporting a 10x sampling speedup with only a 5.6% task-performance drop when cutting DDIM steps to 10% of the original count — flagged by the survey as an isolated, task-dependent data point rather than a general result.

**Real-world vs. simulation split:** most surveyed methods are evaluated in both sim and real-world settings, trained directly on real-world data; a smaller number train exclusively in simulation and transfer zero-shot via domain randomization or scene reconstruction; a mostly-RL subset is evaluated only in simulation.

## Limitations (as stated by the authors)

Two, both explicit in Section 6.1:

1. **Generalizability.** Most trajectory-generation methods rely on imitation learning / behavior cloning, inheriting dependence on demonstration data quality and diversity and the covariate-shift problem for out-of-distribution situations. Offline-RL methods share the same fundamental dependence on existing data coverage and are similarly unable to react to distribution shift, while also needing more careful tuning to avoid overfitting than imitation learning does. Data-augmentation approaches mostly only touch visual appearance (colors, textures, backgrounds, distractors) rather than trajectories/actions themselves, which the authors say limits how much generalizability augmentation alone can buy. VLA integration helps with multi-task/long-horizon generalization but "often lacks action precision," which is exactly why diffusion-based refinement heads get bolted onto VLAs in several surveyed methods.
2. **Sampling speed.** The iterative denoising process is inherently slow, which the authors call the "principal limitation" of DMs. DDIM remains dominant in practice despite DPM-solver reportedly performing better, because DPM-solver's advantage has only been demonstrated on image-classification benchmarks — not validated for robotic manipulation specifically. The authors explicitly flag that very few works quantify the performance-vs-speed trade-off of reduced sampling steps for robotics; they cite exactly one (Ko et al., 2024) with a concrete number.

No other limitations are claimed by the authors in that section; anything below is this report's own read, not theirs.

## Critical read

**The "first survey" framing is a claim to take at face value, not verify.** [Guessing — I have no way to confirm novelty priority from inside the paper itself] The authors state this plainly ("to the best of our knowledge, we provide the first survey..."), which is a standard and reasonable hedge, but it's their self-assessment of the literature landscape, not something this report can independently confirm.

**The survey's headline comparison numbers inherit the same construction problems as the papers they're pulled from — and the survey doesn't re-flag them.** The "3D-DP beats DP by 24.2%" and "by 50% in the real world" figures are restated directly from 3D Diffusion Policy's own paper, with no caveat about how those baseline numbers were selected or whether the comparison controls for training data, compute, or action-space choice — the same category of methodological question we flagged for Diffusion Policy's own 46.9% headline number in the companion report. This survey is a second-order aggregation of first-order numbers that were already somewhat construction-dependent; it doesn't add scrutiny at that second layer, it just relays it.

**Depth is very uneven between Section 2 and Section 4.** The mathematical preliminaries (Section 2) are genuinely derived — full equations for SMLD and DDPM, the forward/reverse processes, the loss functions. The applications section (Section 4), which is most of the paper by length, is almost entirely citation-classification prose: "Method X does Y, Method Z does W, Method V combines both." Very few passages explain *how* any specific downstream method's math actually departs from the Section 2 equations — the survey tells you which bucket each method belongs in far more often than it tells you what's mechanically different about the method within that bucket. That's consistent with the paper's explicitly stated goal (systematic classification via taxonomy), but it means this survey is a strong index into the literature and a shallow substitute for reading any individual method's own paper.

**One of the survey's own admissions is arguably its most useful sentence.** Section 6.1.2 states plainly that "only a few evaluations exist that compare DDPM-based, DDIM-based, or other samplers for robotic manipulation, and further investigation is required" — this is the rare moment where the survey identifies a real, specific, checkable gap in the field's evidence base rather than just cataloging what exists. Worth more weight than most of the taxonomy sections, precisely because it's a claim about what's *missing* rather than what's present.

**Grasp-synthesis coverage (Section 4.2) is more mathematically substantive than trajectory coverage, likely reflecting the authors' own research focus.** [Likely — inferable from institutional affiliation and the relative technical depth given to SE(3)-equivariance, bi-equivariance, and the manifold-diffusion problem] The section on diffusing over SE(3) grasp poses goes into real technical depth (the HH⁻¹ = I constraint, Lie-algebra score matching, bi-equivariance), noticeably more than the trajectory-generation section affords to any comparably specific technical mechanism. This isn't a flaw, but it does mean the survey's depth is not evenly distributed across its own stated three application areas — grasp synthesis gets closer to a genuine technical review, trajectory generation gets closer to an annotated bibliography.

## Reference

Wolf, R., Shi, Y., Liu, S., and Rayyes, R. (2025). *Diffusion Models for Robotic Manipulation: A Survey.* arXiv preprint arXiv:2504.08438v3 [cs.RO], July 15, 2025.

## Background: Borrowed Concepts

Citations this survey leans on as building blocks without explaining the mechanism — flagged by the reference-lookout pass, approved for write-up. Each entry states confidence plainly.

### DiT (Peebles and Xie, 2023)
[Certain] Diffusion Transformers replace the U-Net backbone conventionally used for image diffusion with a plain transformer operating over patch tokens: an image (or feature map) is chopped into fixed-size patches, each patch embedded as a token, and the diffusion timestep is injected not via cross-attention or token concatenation but via *adaptive layer norm* (adaLN, later adaLN-Zero) — the timestep (and any class/condition embedding) is used to predict per-channel scale-and-shift parameters that modulate the layer norm inside each transformer block, with residual branches zero-initialized so early training behaves close to an identity function. The result showed that standard transformer scaling laws — bigger model, more tokens, more compute — carry over directly to diffusion without any convolutional inductive bias at all. In this survey's tables, "DiT" as a denoiser choice typically means this adaLN-conditioned patch-transformer design (adapted from image patches to action/observation tokens), which is a specifically different mechanism from the plain cross-attention transformer Diffusion Policy uses — the two get grouped together informally as "transformer-based" in Tables 2, 4, and 5, but they condition on side information in different ways.

[↩ back to Architecture](#architecture)

### FiLM (Perez et al., 2018)
[Certain] FiLM (Feature-wise Linear Modulation) injects side information into a convolutional network without concatenating it into the input. At each layer, a small side network maps the conditioning signal to a per-channel scale γ and shift β, and every channel of that layer's feature map is transformed as γ·x + β before continuing through the network. It's cheap, applies uniformly at every layer, and re-injects the conditioning signal's influence at every depth rather than only once at the input — which matters for a diffusion model that needs the observation to steer denoising consistently across many layers and many denoising steps. This is the same mechanism flagged in the companion Diffusion Policy report; Diffusion Policy is in fact the origin point this survey cites for adapting FiLM into the temporal U-Net backbone.

[↩ back to Architecture](#architecture)

### GIGA Framework (Jiang et al., 2021)
[Guessing — I know the general grasp-detection-in-clutter literature at a conceptual level, but I don't have detailed, confident recall of this specific paper's architecture; treat this as a rough orientation, not a reliable technical account] GIGA ("Synergies between affordance and geometry: 6-DoF grasp detection via implicit representations") frames grasp detection as jointly learning an implicit scene representation — predicting occupancy/geometry at continuous query points, in the style of occupancy networks — together with a grasp-affordance function defined over that same continuous 3D space, trained jointly so the grasp predictor can exploit shared geometric features rather than working from raw points alone. The load-bearing idea the survey borrows the name for is this shared-representation trick: treating "where can I grasp" and "what does the scene's geometry look like" as one learning problem rather than two separate pipelines. Song et al. (2024b), as cited in this survey, take this general conditioning structure (grasp pose conditioned on location plus volumetric features) and replace GIGA's original direct regression with a diffusion model on top. Given the lower confidence here, if GIGA's actual implementation details matter for your purposes, that's a good candidate for running paper-deep-dive on Jiang et al. (2021) directly rather than relying on this gloss.

[↩ back to Architecture](#architecture)
