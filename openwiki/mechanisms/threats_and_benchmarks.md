# Comprehensive Safety Benchmark and Threat Landscape
**Source:** li2026-vla-safety-report.md (Sections 3–8)

This page aggregates the diverse metrics, datasets, and threat models used to evaluate VLA safety across the literature. Because "safety" is not a single metric, this section categorizes risks by their source or nature of testing.

## 🔬 Benchmark & Metric Categories
The field utilizes multiple specialized benchmarks—each designed to probe a specific failure mode:

| Benchmark Name | Primary Focus / Domain | Key Vulnerability Tested | Sample Metric | Source Reliability Note |
| :--- | :--- | :--- | :--- | :--- |
| **VLA-Risk** (296 scenarios) | General hazard detection; complex task failures. | Identifying prohibited/hazardous actions in open environments. | Safety Violation Rate, Success Rate (across difficulty tiers). | Highly cited; useful for general benchmarking. |
| **SafeAgentBench** (750 tasks) | Explicitly hazardous instructions & failure modes. | Agent compliance and refusal capabilities when prompted to perform dangerous tasks. | Rejection Rate (low rate indicates high safety). | Crucial for testing ethical boundaries. |
| **VLA-Arena / VLABench** | Comprehensive task completion combined with safety goals. | Balancing goal achievement vs. risk mitigation across a full episode rollout. | Safety–Performance Pareto frontier, CostNav's Net Value cost-revenue formula. | Good for system-level evaluation. |
| **LIBERO-PRO** | Structured manipulation tasks (tabletop/real world). | Detecting "Success by Memorization" vs. generalization failure. | Reported accuracy rates under perturbation testing. | Used to prevent superficial success claims. |
| **BadRobot / RoboPAIR** | Adversarial querying and prompt injection. | Jailbreaking the model via malicious language inputs or visual perturbations. | Attack Success Rate (ASR). | Critical for LLM/VLA boundary failures. |

## 🧪 Specific Threat Vectors
Understanding the threat model is as important as understanding the defense mechanism:

1.  **Training-Time Backdoors / Poisoning:** The attack involves subtly poisoning the dataset ($\mathcal{D}$) used for fine-tuning (e.g., LoRA training). This can implant a hidden trigger that causes catastrophic failure when presented with specific, rare input sequences—even if the model passes general safety testing.
    *   **Defense Focus:** Data provenance tracking; secure training pipelines.
2.  **Inference-Time Adversarial Perturbations:** These are real-time attacks on the inputs:
    *   **Visual Attacks:** Placing physical patches or using specific lighting to fool the visual encoder (e.g., making a "STOP" sign appear as "GO").
    *   **Language Attacks:** Crafting prompts that exploit semantic loopholes in the VLA's reasoning backbone (e.g., indirect instructions).
3.  **Sim-to-Real Gap:** The inability to guarantee safety metrics achieved in simulation ($\text{Sim}$) hold true when deployed on physical hardware ($\text{Real}$). This is a major unresolved limitation for all current safety claims, as many benchmarks are simulation-only.

## 🚧 Future Research Focus (Unsolved Problems)
*   **Certified Robustness:** Extending formal methods that certify robustness in image classification to the complexity of multi-modal, real-time VLA control policies.
*   **Physical World Transferability:** Developing metrics and methodologies that bridge the gap between simulated safety guarantees and guaranteed physical outcomes.

Developers should always treat any stated "safety rate" or "robustness metric" as a *claim from the original study authors*, not an established consensus truth, until independently verified on physical hardware.