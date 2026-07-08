# OpenWiki Documentation Plan

## Intended Wiki Pages
1.  `/openwiki/quickstart.md`: The main entry point, repository overview, and navigation hub.
2.  `/openwiki/architecture/`: To describe the overall flow of a VLA agent and where safety components integrate.
3.  `/openwiki/domain/safety_concepts.md`: Explaining the core concepts (physical vs. text hazards, profanity analogy).
4.  `/openwiki/mechanisms/inference_time_filtering.md`: Detailed guide on running real-time filters (VLESA, Attention-Guided Filter).
5.  `/openwiki/mechanisms/policy_alignment.md`: Details on constrained training methods (SafeVLA, CMDPs).
6.  `/openwiki/mechanisms/world_modeling_safety.md`: Explaining predictive safety using world models (SafeDojo, World-Env).

## Source Evidence & Grounding
*   **Source:** `/VLA-safety/README.md`
    *   This single file provides the conceptual foundation, outlining three parallel approaches to VLA safety: inference-time filtering, constrained policy alignment, and world model prediction.
    *   It grounds all technical sections (Mechanisms).

## Remaining Questions / Ambiguities
*   The current repository structure is extremely focused on documentation of *concepts* rather than code implementation. I assume the goal is to document the state-of-the-art research landscape captured in the README, providing architectural and conceptual guidance for future development/research agents.
*   No source files were found outside `/VLA-safety/` that represent core functional code (e.g., `main.py`, etc.). The documentation must focus heavily on linking theory to practice within this single domain.

## Plan Completion:
I will create the initial 7 pages listed above and ensure they link coherently from `quickstart.md`. I will start with writing `/openwiki/quickstart.md`.