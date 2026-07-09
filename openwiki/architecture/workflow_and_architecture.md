# ⚙️ VLA Agent Safety Operational Workflow (The Safety Loop)

This document defines the canonical, end-to-end safety process that must be executed for any action proposed by a Vision-Language-Action (VLA) agent in an embodied or highly constrained digital environment. The system shifts from simply *checking* safety to enforcing it through a mandatory multi-layered pipeline.

## 🗺️ The Agent Action Pipeline Diagram
(Conceptual Flow: Prompt $\rightarrow$ Planning $\rightarrow$ Safety Gates $\rightarrow$ Execution)

1.  **Input:** User Prompt & Current State (Vision Context)
2.  **Planner/Core Model:** Generates a high-level action plan or candidate sequence of actions $A_{candidate}$.
3.  **Safety Gate Stack (Mandatory):** The proposed action $A_{candidate}$ is passed sequentially through three mandatory, specialized safety modules:

    *   ✅ **Policy Alignment Check (Conceptual/Pre-computation)**: Does this action align with known safe policies?
    *   🔮 **World Model Prediction (Simulation)**: What does executing $A_{candidate}$ *predictively* do in a simulated environment? Is the outcome dangerous or impossible?
    *   🛡️ **Inference-Time Filter (Real-time/Constraint)**: Does this action violate immediate, hard constraints (e.g., "do not grasp glass," "never move faster than $X$")?

4.  **Output:** The validated, executable Action Sequence $A_{validated}$, or a Safety Failure Report if blocked at any gate.

---

## 🛠️ Detailed Operational Steps and Gate Logic

### Step 1: Core Planning (The Origin)
*   **Component:** VLA Core Model / Planner
*   **Input:** Prompt ($\mathbf{P}$), Current State ($\mathbf{S}_t$), History ($\mathbf{H}$)
*   **Output:** Candidate Action Sequence ($A_{candidate}$).

### Step 2: Safety Gate Stack Execution (The Constraints)
The gate stack is executed in a specific, ordered sequence. Failure at any single step immediately halts the process and triggers a **Safety Failure Report**, forcing the Planner to re-prompt or adjust its plan.

#### A. Policy Alignment Check (Conceptual Vetting)
*   **Goal:** Ensure the entire planned behavior adheres to pre-trained safe policies defined by human preference or formal methods ($\pi_{safe}$).
*   **Mechanism Link:** [openwiki/mechanisms/policy_alignment.md](openwiki/mechanisms/policy_alignment.md) (SafeVLA, CMDPs).
*   **Pseudo-Logic:** Check if $A_{candidate}$ significantly deviates from the expected safe policy space $\mathcal{P}_{safe}$.

#### B. World Model Prediction (Simulation Vetting)
*   **Goal:** Predict potential *future* states ($\mathbf{S}_{t+1}, \dots, \mathbf{S}_{t+k}$) resulting from $A_{candidate}$ in a simulated environment. This is crucial for avoiding indirect hazards (e.g., knocking over an object that then falls).
*   **Mechanism Link:** [openwiki/mechanisms/world_modeling_safety.md](openwiki/mechanisms/world_modeling_safety.md) (SafeDojo, World-Env).
*   **Failure Condition:** If the predicted trajectory leads to a state $\mathbf{S}_{t+k}$ categorized as high-risk (e.g., collision probability $> X\%$), the action is blocked.

#### C. Inference-Time Filter (Hard Constraints)
*   **Goal:** Apply immediate, hard constraints and guardrails on the *actual command* being executed at time $t$. This acts as a final, fail-safe check against low-level violations that might bypass higher-level planning logic.
*   **Mechanism Link:** [openwiki/mechanisms/inference_time_filtering.md](openwiki/mechanisms/inference_time_filtering.md) (VLESA).
*   **Failure Condition:** If the command violates a known physical or policy constraint, it is blocked immediately.

### Step 3: Execution
If $A_{candidate}$ passes all three gates, it becomes $A_{validated}$ and is executed in the real world/system environment.

---

## 📚 Source Map & Next Steps for Developers

*   **Reference Architecture:** This document provides the *sequence*. The individual mechanism pages provide the *details* of each gate.
*   **Implementation Focus:** When building a safety layer, focus on making the output format predictable: either a `PASS` signal or a structured `SafetyFailureReport` containing the reason and remediation suggestion (e.g., "Violation Type: Collision; Location: [X,Y,Z]").

**To start implementation work in this domain, begin by reviewing:**
1.  [openwiki/mechanisms/policy_alignment.md](openwiki/mechanisms/policy_alignment.md)
2.  [openwiki/mechanisms/world_modeling_safety.md](openwiki/mechanisms/world_modeling_safety.md)

***This workflow page should be linked prominently in the Quickstart Guide and Architecture section.***