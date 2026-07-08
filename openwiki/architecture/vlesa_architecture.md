# VLESA Architecture Documentation

This section details the Vision-Language-Action Safety Assessment (VLESA) architecture, designed to ensure safety throughout the vision-language-action process by integrating safety checks directly into the action planning workflow.

## Core Components
The VLESA architecture is built around several interconnected modules:

*   **Vision Encoder:** Responsible for processing raw visual inputs into a meaningful representation.
*   **Language Model (LM):** Processes textual instructions and context to understand user intent.
*   **Action Policy Module:** Translates the combined vision and language understanding into potential executable actions, incorporating initial safety constraints.
*   **Safety Critic Module (SCM):** The core safety mechanism that assesses proposed actions against a comprehensive set of safety policies.
*   **Alignment Layer:** Manages the dynamic alignment between the model's generated intent and the established safety boundaries.

## Workflow
The operational flow of VLESA is structured as a multi-stage safety check integrated directly into the action planning process:

1.  **Input Phase:** Vision/Language input is processed by the Vision Encoder and Language Model.
2.  **Action Proposal:** The model proposes an action via the Action Policy Module, which incorporates preliminary safety checks.
3.  **Safety Assessment:** The proposed action is passed to the **Safety Critic Module (SCM)** for rigorous evaluation against predefined safety policies.
4.  **Decision & Refinement:** If the SCM approves the action, it proceeds. If not, the system initiates a feedback loop to re-align the model's intent, ensuring safe outcomes.

This architecture emphasizes a proactive, multi-stage safety check to prevent unsafe actions before execution.

*For detailed findings on safety assessments, please refer to the [VLA Safety Report](/VLA-guardrails/li2026-vla-safety-report.md).*