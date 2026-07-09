# Core Safety Concepts and Terminology

This document serves as the canonical glossary and conceptual guide for the VLA safety project. Understanding these foundational terms is essential for reading, developing, or modifying any part of the system.

## 📖 Key Definitions
**VLA (Vision-Language-Action):** A model that takes multimodal input (visual observations, language instructions) and outputs a sequence of physical actions suitable for execution by an agent or robot.

**Alignment:** The state where the AI system's goals, behaviors, and reward function match the intentions and values of its human operators. Poor alignment leads to "specification gaming."
*   *Related:* **Policy Alignment**.

**Safety Guardrail (or Mechanism):** A defined architectural component that intercepts, validates, or modifies a model's raw output based on hard constraints, ethical guidelines, or physical laws. They enforce safety redundancy.

**Goal-Conditioned Safety:** The approach of framing the task not just as "achieve goal G," but as "achieve goal G *while maintaining constraint C*." This elevates safety from an afterthought to a primary optimization objective.
*   *Related:* **World Modeling Safety**.

**Specification Gaming (or Goal Misalignment):** A failure mode where the model finds a loophole in the stated requirements to achieve maximum score or completion, but in doing so, violates unstated ethical constraints or common sense rules (e.g., fulfilling "open the box" by breaking it if that is faster).
*   *Mitigated by:* **Policy Alignment** and **World Modeling Safety**.

## 📚 Concepts Mapped to Mechanisms
| Concept | Description | Primary Mechanism Responsible | Source Area |
| :--- | :--- | :--- | :--- |
| **Robustness** | Ability to maintain performance under noisy, adversarial, or out-of-distribution inputs. | Inference Time Filtering (ITF) | `openwiki/mechanisms/` |
| **Redundancy** | Using multiple, independent checks on a single critical decision. | Dual-Loop Defense | `openwiki/mechanisms/` |
| **Feasibility** | Ensuring the predicted action can physically occur within the environment's constraints. | World Modeling Safety | `openwiki/mechanisms/` |

***Usage Note:** When integrating new safety logic, consult this document to ensure correct terminology and proper placement within the existing conceptual framework.*