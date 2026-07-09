# Safety Alignment Architecture Comparison

## 📊 Overview and Comparative Analysis
Since this repository contains multiple pioneering works on robot safety, understanding *how* they differ is more critical than knowing their individual components. This comparison matrix helps agents select the appropriate methodology based on the primary research goal (e.g., focusing on intent, formal guarantees, or data efficiency).

| Feature | VLESA (VLM Filtering) | SafeVLA (CMDP/Lagrangian) | World-Env (World Model RL) | SafeDojo (Interactive Modeling) |
| :--- | :--- | :--- | :--- | :--- |
| **Core Technique** | Goal-Conditioned Filtering & VQA Scoring | Constrained Optimization (CMDP, Lagrangian Relaxation) | Simulation & Continuous Signal Feedback | World Model + Dual-Branch Cost/Reward Head |
| **Primary Focus** | **Intent ($\hat{g}$)**: Distinguishing safe vs. unsafe *given* the wearer's goal. | **Formal Guarantees**: Mathematically bounding policy actions to satisfy defined safety predicates. | **Data Efficiency & Termination**: Overcoming sparse rewards and physical simulation limitations. | **Safety Imagination**: Generating diverse, dangerous failure scenarios (rollouts) to train on proactively. |
| **How Safety is Applied** | Filters candidates: Calculates $Q_{\varphi}(I, a, \hat{g})$. | Optimizes policy constraints: $\max J_r$ subject to $J_{c_i} \le b_i$. | Signals termination/cost: Uses $R(o, g)$ to stop rollouts; cost is secondary. | Penalizes imagined risks: Uses two independent heads ($f_\text{task}$ and $f_\text{safe}$) in a constrained RL update. |
| **Key Assumption** | The VLM can accurately infer the user's latent goal $\hat{g}$. | Safety constraints can be mathematically formalized as cost functions ($\Phi, \Psi$). | A learned world model can accurately predict future visual frames and task completion probability. | The generated failure modes (Corner, Blind Spot) are comprehensive enough to cover real-world risk. |
| **Output Signal** | $\text{Score}(a_k)$ (Actionable intervention). | Lagrangian Multiplier update ($\lambda$) and policy constraint adherence check. | Continuous Probability $R \in [0, 1]$ (Termination signal). | Composite Advantage $\hat A^\text{safe}$ (Constrained RL objective). |
| **Best Use Case** | Real-time human monitoring, assistive robotics with user intent context. | High-stakes environments where formal proof of safety bounds is necessary (e.g., industrial robots). | Tasks requiring complex long-horizon planning or few real-world demonstrations (data scarcity). | Training robust policies by forcing them to confront potential failure modes they might never see in standard data. |

### 🧭 Comparative Summary for Agents
*   **Choose VLESA** when the *intent of the human operator* is the most critical variable, allowing safety scoring to be goal-relative.
*   **Choose SafeVLA** when the system requires a high degree of **provable, formal safety guarantees**, treating risk as a quantifiable physical constraint.
*   **Choose World-Env** when your primary bottleneck is **data volume or simulating rare failure events**, relying on the power of world models and continuous reward signals.
*   **Choose SafeDojo** when you want to systematically **test the boundaries of possibility**, augmenting training data by imagining failure modes that are hard to capture otherwise.

***Note: These architectures are not mutually exclusive. A sophisticated agent might combine elements, such as using World-Env's simulator to generate diverse inputs for SafeVLA's CMDP training set.***