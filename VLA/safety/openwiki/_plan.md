# OpenWiki Documentation Plan

**Goal:** Initialize comprehensive technical and business documentation for the VLA safety project repository in `/openwiki/`.

**Approach:**
1.  Create `quickstart.md` as the entry point, providing a high-level overview.
2.  Establish core sections based on observed themes: Architecture, Safety Mechanisms, Concepts (Domains), Testing/Evaluation, and Operations/Workflow.
3.  Synthesize documentation from source files, academic papers (PDFs/Reports found in git history), and existing READMEs to explain *why* components exist and how they relate to the overall safety goals of VLA models.

**Intended Wiki Pages:**
1. `openwiki/quickstart.md` (Overview & Roadmap)
2. `/openwiki/architecture/vlesa_architecture.md` (Specific high-level architecture component)
3. `/openwiki/mechanisms/safety_overview.md` (General safety approach)
4. `/openwiki/mechanisms/dual_loop_defense.md` (Detailed mechanism 1)
5. `/openwiki/mechanisms/inference_time_filtering.md` (Detailed mechanism 2)
6. `/openwiki/mechanisms/policy_alignment.md` (Detailed mechanism 3)
7. `/openwiki/domains/safety_concepts.md` (Core concepts and terminology)
8. `/openwiki/testing/evaluation_strategy.md` (How models are tested and evaluated)

**Source Evidence Focus:**
*   Git history points heavily to VLA safety reports, diffusion planning, and formal defense mechanisms (e.g., `VLA-guardrails/`, `VLA/Diffusion-robotics/`).
*   The existing structure in `/openwiki/` already suggests these sections (`architecture/`, `mechanisms/`, etc.) are the right targets.

**Next Steps:** Write content for these files, starting with `quickstart.md`.