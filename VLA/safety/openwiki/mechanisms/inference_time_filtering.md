# Inference Time Filtering (ITF)

Inference Time Filtering (ITF) is a critical class of guardrails that operate *after* the VLA model has generated an intended action or predicted trajectory, but *before* the action is executed by the robot/agent. It acts as a real-time sanity check on the raw output stream.

## 🎯 Purpose
The goal of ITF is to filter out "safe-looking" but fundamentally unsafe outputs. Unlike pre-planning checks which constrain the search space, ITF examines the generated candidate $A'$ and decides if it violates known constraints (e.g., kinematic limits, proximity rules) or predicted failure modes.

## 🛠️ Technical Implementation
ITF typically involves running the predicted action $A'$ through several lightweight, fast validation modules:

1.  **Constraint Checking:** Does $A'$ violate hard physical limits ($|A'| < A_{max}$)? (Basic check).
2.  **Predictive Simulation:** Running a micro-simulation step using $A'$ to predict the immediate next state $S'$. If $S'$ results in an unsafe or unrecoverable state, $A'$ is discarded.
3.  **Semantic Check:** Does $A'$ align with the high-level intent? (e.g., if the instruction was "pick up the cup," and $A'$ points toward the floor, it fails this check).

## 🧠 Relation to other Mechanisms
ITF often complements Dual-Loop Defense. While $\pi_{safety}$ might reject a policy altogether, ITF can catch subtle violations in a seemingly valid action sequence that passed the planning phase.

***Use Case:** This is ideal for catching momentary deviations or minor kinematic slips not accounted for in the core model's training distribution.*