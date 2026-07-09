# Vision-Language Embodied Safety Agent (VLESA) Architecture

## 🎯 Purpose
VLESA is a system designed to act as a real-time, goal-conditioned safety monitor for egocentric (first-person) video streams. Its primary innovation is treating safety not just as an action filter, but as a constraint that must be explicitly conditioned on the inferred **intent** or **goal ($\hat{g}$)** of the wearer.

## ⚙️ Core Methodology
VLESA operates as a three-stage pipeline, composing several existing Vision-Language Models (VLMs) rather than being a single end-to-end network. The key difference from previous works is that the goal $\hat{g}$ is explicitly factored into the prediction process.

### 1. Intent–Action Prediction Agent
*   **Input:** Stream of egocentric frames ($I_{1:t}$).
*   **Model Used:** A frozen, off-the-shelf VLM (e.g., Llama-4-Scout-17B-16E-Instruct).
*   **Output:** Jointly infers the latent task goal ($\hat{g}$) and proposes $K$ candidate next actions ($a_k$), expressed in a structured scene-graph triplet format: `(subject, predicate, object)`.
*   **Key Equation:** The probability is factored as $P(\hat{g}, a_{t+1} | I_{1:t}) = P(\hat{g} | I_{1:t}) \cdot P(a_{t+1} | I_{1:t}, \hat{g})$. This factorization makes $\hat{g}$ an exposed variable for safety scoring.

### 2. Goal-Conditioned Safety Q-Filter
*   **Purpose:** To score candidate actions against the inferred goal $\hat{g}$.
*   **Model Used:** A small, purpose-trained VLM (Qwen3-VL-2B-Instruct).
*   **Mechanism:** It answers a Visual Question Answering task: given an image $I$, the goal $g$, and action sentence $a$, it outputs "Safe" or "Unsafe" and provides Chain-of-Thought reasoning.
*   **Safety Metric:** The output is mapped to a scalar $Q_{\varphi}(I, a, g) \in \{-1, +1\}$, where negative values indicate safety (the boundary is zero). This mirrors the concept of Control Barrier Functions (CBF).

### 3. Constrained Decoding & Intervention
*   **Action:** Each candidate action $a_k$ receives a composite score: $\text{Score}(a_k) = (1 - k/K) + \alpha \cdot (-s_k)$, where $s_k$ is the safety score from Step 2.
*   **Intervention Logic:** The system filters for candidates $S$ that are judged safe ($s_k < \tau$). If $S$ is non-empty, it outputs the highest-scoring candidate in $S$. If *all* candidates are unsafe, it alerts danger and falls back to the least unsafe alternative.

## 📚 Data & Training
*   **Dataset:** **EgoSafety**, built on the existing Ego4D corpus using Egocentric Action Scene Graph (EASG) annotations.
*   **Unsafe Data Generation Trick:** Instead of expensive video rollouts, unsafe examples are generated cheaply by fixing the *image frame and goal* and prompting a VLM to produce a contextually plausible **unsafe variant of the scene-graph triplet**. This converts an expensive video problem into a graph-editing/VQA labeling problem.
*   **Training Method:** The Q-filter is trained using **Group Relative Policy Optimization (GRPO)**—a lightweight Reinforcement Learning technique that computes group-relative advantages to reduce variance without requiring a separate critic network.

## 📊 Performance & Claims
*   On the ASIMOV-2.0-Video benchmark, VLESA more than doubles intervention accuracy over baseline methods and beats multiple frontier foundation models (GPT-5, Claude, Gemini) on the accuracy/timeliness Pareto front.