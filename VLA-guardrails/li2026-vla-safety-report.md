# Vision-Language-Action Safety: Threats, Challenges, Evaluations, and Mechanisms

## TL;DR

This is a survey (not a new model or attack) that maps the emerging field of safety for Vision-Language-Action (VLA) models — robot policies that fuse a visual encoder, a language-model backbone, and an action decoder into one network. The authors argue VLA safety is qualitatively different from text-only LLM safety because failures produce irreversible physical consequences, the attack surface spans vision/language/proprioceptive-state channels, defenses are latency-constrained, and errors compound over long action horizons. The paper's organizing device is a **two-axis taxonomy**: attack timing (training-time vs. inference-time) crossed with defense timing (training-time vs. inference-time). It surveys training-time backdoors and data poisoning, inference-time jailbreaks/perturbations/physical attacks, the corresponding defenses (safe RL, guardrails, runtime monitors, physical fail-safes), the fragmented benchmark/metric landscape, six real-world deployment domains, and open problems. It claims to be the first comprehensive survey unifying this literature.

## Architecture

Since this is a survey rather than a model paper, "architecture" here means two things: (1) the generic VLA architecture the survey assumes as background, and (2) the survey's own organizing taxonomy.

**Generic VLA architecture (Section 2.2, Figure 3).** Modern VLA systems share three components, each named as a distinct threat surface:
- **Visual encoder** — maps camera frames into patch embeddings (CLIP-style ViT or SigLIP), usually frozen/lightly fine-tuned during robot learning. Flagged as the primary entry point for adversarial image perturbations.
- **Language backbone** — a large autoregressive transformer (e.g., a LLaMA-family model) that fuses projected visual tokens with the tokenized instruction and reasons jointly. Because it inherits a general-purpose LLM, it also inherits the LLM vulnerability surface (prompt injection, adversarial suffixes, iterative jailbreak refinement).
- **Action decoder** — turns the backbone's output into executable control. Three paradigms are compared: token-based autoregressive decoding (RT-2, OpenVLA), continuous regression/diffusion heads (Octo), and flow matching (π0, π0.5). The paper links each choice to a distinct safety profile — e.g., token decoding is slower per action and its softmax output competes for probability mass between "safe refusal" tokens and unsafe action tokens (Eq. 7), while action-chunked continuous/flow decoders create an "intra-chunk visual blind spot" (Section 2.4, exploited later by SilentDrift).

Formally, the survey frames VLA control as a POMDP with observation $o_t=(v_t,s_t)$ (vision + optional proprioceptive state) and policy $\pi_\theta(a_t\mid o_{\le t}, l)$ trained via behavior cloning (Eq. 1–2). This formalism is used explicitly to locate where attacks can enter: corrupting the training set $\mathcal{D}$ (training-time attacks) vs. manipulating $o_t$ or $l$ at deployment (inference-time attacks).

**The survey's own taxonomy (Figure 2, restated throughout).** Two independent timing axes:
- Attack timing: training-time (data poisoning, backdoors) vs. inference-time (jailbreaks, perturbations, physical interventions).
- Defense timing: training-time (alignment/optimization/human feedback) vs. inference-time (guardrails, runtime monitoring, physical fail-safes).

Every subsequent section (3–5) is slotted into one of these four quadrants, and Section 6–8 sit "on top" as evaluation and forward-looking synthesis. The dual-loop inference-time defense architecture (Figure 7) is the one genuinely diagrammable *mechanism* in the paper: a high-frequency (~100Hz) "Fast Reflexes" loop that is semantically blind but physically strict (control barrier functions, kinematic projection, impedance control), paired with a low-frequency (~1Hz) "Slow Reasoning" loop that uses VLMs/LLMs to translate natural-language constraints into formal logic (e.g., Signal Temporal Logic) and monitor execution, feeding back the safe region $\Omega_{safe}$ that the fast loop enforces.

## Data

The survey doesn't introduce a dataset; it discusses the data landscape that training-time attacks/defenses operate on:
- **Open X-Embodiment** — ~1M demonstrations across 22 embodiments from 21 institutions; the dominant fine-tuning corpus for generalist VLA policies (used by OpenVLA at 970k episodes, Octo at ~800k trajectories).
- **LIBERO / LIBERO-PRO** — structured tabletop manipulation benchmark; LIBERO-PRO is highlighted as a corrective work showing the original LIBERO's reused train/eval configurations let models "succeed" via memorization rather than generalization (reported >90% accuracies called misleading).
- **Poisoned/backdoor training data** as an attack surface in its own right: the survey stresses that VLA fine-tuning data is typically collected via teleoperation or crowdsourced pipelines with little vetting, and that under a "Training-as-a-Service" LoRA fine-tuning threat model, a user-submitted poisoned dataset can implant a backdoor without touching the frozen base weights.
- **Safety benchmark corpora** (treated as "data" for evaluation rather than training): VLA-Risk (296 scenarios / 3,784 episodes), SafeAgentBench (750 tasks / 10 hazard categories), AgentSafe (45 scenarios / 1,350 hazardous tasks / 9,900 instructions), VLA-Arena (170 tasks), VLABench (100+ categories / 2,000+ objects), BadRobot's query set (230 malicious queries). These are cataloged in Table 5.

No discussion of demographic bias, licensing, or consent issues in the underlying robot datasets is given — the survey's data lens is entirely about integrity/poisoning, not representativeness.

## Training setup

Section 2.3 lays out the training paradigm stages that later sections attach threats/defenses to:
1. **Vision-language pretraining** on web-scale image-text/video data (inherited biases/misinformation noted as a pass-through risk).
2. **Robot demonstration fine-tuning** via the behavior-cloning objective (Eq. 2) — the stage where data poisoning/backdoors are injected (Section 3).
3. **Preference alignment** — RLHF-style or DPO-style adaptation to embodied rollouts, flagged as harder than text RLHF because physical rollout reward signals are expensive and unsafe exploration has real consequences.
4. **Parameter-efficient adaptation (LoRA)** — called out specifically as a training-time attack vector, since a Training-as-a-Service LoRA pipeline can be poisoned by a submitted dataset without full-parameter access.

Training-time *defenses* (Section 4) are organized as three families, each with a stated mechanism:
- **Data/perception/reward-centric alignment** — e.g., EvoVLA's Stage-Aligned Reward (stage dictionaries + triplet contrastive learning to avoid "stage hallucination"); Safe-Night VLA's frozen RGB backbone with added thermal/depth input and asymmetric photometric augmentation.
- **Policy-centric safety optimization** — SafeVLA formulates alignment as a constrained MDP (Eq. 5) with an explicit cost budget per safety constraint; SORL uses a learned safety critic; VLA-Forget performs post-training unlearning of unsafe associations; Hi-ORS filters trajectories via reward-based rejection sampling.
- **Human-in-the-loop refinement** — APO relabels human corrective interventions as preference pairs (Eq. 6, a DPO-style loss with adaptive per-sample weights); Hi-ORS (again) treats interventions as a filtering signal rather than a preference signal.

No compute budgets, hardware, or hyperparameters are reported anywhere in the survey (expected, since it aggregates dozens of external papers rather than running its own experiments).

## Evaluation & results

The survey doesn't run its own evaluation; it synthesizes the benchmark/metric landscape (Sections 5.3, 6) and reports numbers *from* the surveyed papers:
- **Benchmark categories** (Table 5): adversarial robustness (VLA-Risk, VLATest), task-level safety (SafeAgentBench, AgentSafe, SafeMind), comprehensive capability+safety (VLA-Arena, VLABench, LIBERO-PRO, CostNav), jailbreak/alignment (BadRobot, RoboPAIR, Shawshank), and runtime-monitoring/semantic-alignment (ASIMOV, SAFE-SMART, SAFE).
- **Metrics** (Table 6), grouped as task-level (Safety Violation Rate, Rejection Rate, Success Rate), behavioral (Collision Rate, multi-stage Safety Score, SPL), robustness (Attack Success Rate, Performance Drop Rate, Certified Robustness Radius — noted as still an open research gap for VLA), and composite/deployment (CostNav's Net Value cost-revenue formula, the Safety–Performance Pareto frontier, temporal persistence of attacks).
- **Headline numbers cited from other papers**, taken at face value: RoboPAIR reports 100% jailbreak ASR across white/gray/black-box driving and quadruped settings; VLATest reports success rates as low as 0.5–12.4% across four manipulation-task difficulty tiers for seven VLA models under fuzzed perturbations; SafeAgentBench reports only ~10% rejection rate for explicit hazardous instructions even from the "most safety-conscious" agent tested.
- **Figure 5** compares Attack Success Rate across LIBERO-Object/Spatial/Goal/10 for OpenVLA and π0 under 8 training-time attack methods — mostly in the 85–100% ASR range, illustrating that current backdoors are not just theoretical curiosities but reliably reproduce across benchmarks and both discretized (OpenVLA) and flow-matching (π0) action decoders.
- **Figure 10** shows an arXiv-publication-count trend (VLA papers vs. VLA-safety papers, 2020–2026, with a "predicted trend" extrapolation) used as motivating evidence that safety attention is scaling with capability research but from a smaller base.

Because none of these numbers were produced by the survey's own authors, they should be read as reported claims from primary sources, not independently verified results.

## Limitations (as stated by the authors)

The authors state their limitations explicitly in Section 9:
- The VLA safety literature is growing rapidly and "new attacks, defenses, and benchmarks will inevitably appear after the present snapshot."
- Their taxonomies are "designed to be forward compatible" but "may require extension as physical attacks and fleet-level threats mature."
- They acknowledge choosing "breadth over depth in places where the underlying research is still nascent," specifically flagging certified robustness, runtime monitoring, and regulatory integration as threads likely to warrant their own dedicated surveys.
- Section 6.1's own discussion of the field notes (as a property of the literature, which the survey inherits) that "metrics remain heterogeneous, adversarial splits are uneven, and simulation dominates at the expense of physical validation" — i.e., most safety claims the survey aggregates have not been validated on physical hardware (Section 7.7, Figure 9's sim-to-real gap discussion).

## Critical read

**What's genuinely a contribution vs. organization of existing work.** This is explicitly a survey, so there's no new attack, defense, or model. Its value proposition is the two-axis (attack-timing × defense-timing) taxonomy plus being — as claimed — the first paper to unify backdoors, jailbreaks, runtime monitoring, benchmark analysis, and six deployment domains under one framework. That's a real organizational contribution if the field really was fragmented as claimed, but the taxonomy itself (training-time vs. inference-time attack/defense) is a fairly standard axis borrowed wholesale from the adversarial-ML and backdoor-attack literature on non-embodied models; the novelty is applying it systematically to VLA rather than inventing a new dimension. The more distinctive framing is the "dual-loop" fast/slow inference-time defense synthesis (Figure 7) and the emphasis on action-chunking as a structurally new attack surface (the "intra-chunk visual blind spot") — those feel like the survey's own conceptual additions rather than restatements of a single source paper.

**Consistency and sourcing.** Nearly every quantitative claim (ASR numbers, success rates, benchmark scales) is attributed to a specific cited paper, which is good survey practice, but it also means the paper's credibility is only as strong as the (often very recent, sometimes unpublished/arXiv-only) primary sources — several citations are dated 2026 and appear to be contemporaneous preprints rather than peer-reviewed, vetted results. A reader should treat headline numbers like "100% ASR" or "0.5% success rate" as claims from the original authors under their own experimental conditions, not as consensus figures.

**Coverage gaps the survey doesn't fully own up to.** The survey is almost entirely manipulation/navigation-robot-centric; despite listing autonomous driving, healthcare, industrial, service, and outdoor domains in Section 7, the technical depth (Sections 3–5) draws its running examples overwhelmingly from tabletop manipulation (OpenVLA/π0/LIBERO). The claim of being a "comprehensive" cross-domain treatment is stronger than the actual balance of material. Similarly, the "Data" dimension of safety (bias, provenance, consent, labor conditions of teleoperated demonstration collection) is essentially absent — the survey's notion of data risk is narrowly about adversarial poisoning, not about the broader data-ethics concerns common in ML-safety surveys of LLMs.

**Open questions the survey correctly flags as unresolved** (Section 8): certified robustness for full trajectories rather than single frames or single images (extending image-classifier certification to sequential, multi-modal, real-time-constrained VLA control is unsolved); the mismatch between studied threat models (digital pixel-space perturbations) and deployed threat models (physically realizable patches, object substitution, lighting); and the sim-to-real transfer of any safety guarantee, since the paper itself notes nearly all cited benchmarks are simulation-only.

## Reference

Qi Li, Bo Yin, Weiqi Huang, Ruhao Liu, Bojun Zou, Runpeng Yu, Jingwen Ye, Weihao Yu, Xinchao Wang. "Vision-Language-Action Safety: Threats, Challenges, Evaluations, and Mechanisms." arXiv:2604.23775v1 [cs.RO], 26 Apr 2026. Project page / living repository: https://github.com/LiQiiiii/Awesome-VLA-Safety
