# Operational Workflows
## Running and Monitoring Policies (Deployment)

This page details the operational considerations required to take a trained diffusion policy from simulation into a real-world robotic control loop. The workflow must account for latency, state drift, and safety overrides.

### 🔄 Standard Inference Cycle
1.  **Observation Acquisition:** A reliable sensor fusion pipeline captures synchronized data (RGB-D, IMU, Joint State) at high frequency ($>30$ FPS). This forms the raw observation $s_t$.
2.  **Preprocessing & Conditioning:** The observed state $s_t$ is encoded into a structured context vector $\mathbf{c}_t$, which includes:
    *   Current joint configuration ($\text{JointAngles}$).
    *   Target goal/waypoint ($g$).
    *   System metadata (e.g., current task ID, safety status).
3.  **Trajectory Prediction:** The core diffusion model $\epsilon_\theta$ generates a noisy action sequence estimate $\hat{\mathbf{a}}_{1:T}$. This requires managing the iterative denoising process with strict timing constraints.
4.  **Filtering & Control:** The raw output $\hat{\mathbf{a}}$ is passed through low-level controllers (e.g., PID or Model Predictive Controllers) that enforce physical limits, kinematics, and immediate safety checks before being sent to the actuators.

### ⚠️ Operational Safety Requirements
The operational flow *must* include a dedicated safety layer that runs parallel to the prediction pipeline. This mechanism performs continuous monitoring:
*   **Out-of-Distribution (OOD) Detection:** Flagging states or actions significantly outside the training distribution.
*   **Hard Constraint Violation:** Checking for collisions, joint limits, or physical impossibilities in $\hat{\mathbf{a}}$.
*   **Fail-Safe Handover:** If any safety mechanism triggers a violation, the system must immediately revert control to a predefined safe state (e.g., stopping all motion and returning to human supervision).

### 🧪 Monitoring & Logging
All operational runs require robust logging:
*   Input State ($\mathbf{s}_t$), Predicted Action ($\hat{\mathbf{a}}_t$), Actual Applied Action ($\mathbf{a}_{actual}$), and the Safety Status flag. This data is crucial for debugging failures, refining reward functions, and understanding model drift.