# Welcome to VLA Safety Documentation

This repository, housed under the `/VLA-safety` module, is dedicated to documenting and organizing the state-of-the-art research and technical approaches for ensuring safety in Vision-Language-Action (VLA) models and embodied AI agents.

Unlike traditional software projects with defined APIs and business logic, this repository serves as a conceptual map of advanced AI safety mechanisms. It translates the abstract safety concerns found in academic literature—such as preventing physical hazards in robotics or malicious actions in LLMs—into actionable architectural patterns for future research and development.

## 🚀 Quickstart Guide

To begin using this documentation:
1. **Understand the Goal:** The primary goal is to address the challenge of "profanity" (i.e., dangerous, unsafe, or prohibited behavior) when an agent operates in a physical or digital environment.
2. **Read the Concepts:** Start by reading [Domain Concepts: VLA Safety Foundations](openwiki/domain/safety_concepts.md) to understand the theoretical parallels between text moderation and embodied action safety.
3. **Explore Mechanisms:** Dive into the core implementation concepts under the [Safety Mechanisms Directory](openwiki/mechanisms/). These sections detail three major types of preventative controls: Inference-Time Filtering, Policy Alignment, and World Model Prediction.

## 🏛️ Architecture Overview
The architecture is not a single monolithic system but a layered set of safety checks applied during VLA operation (as described in the [Architecture Notes](openwiki/architecture/index.md)). The goal is to shift from reactive monitoring to proactive constraint enforcement at multiple points in the agent pipeline.

## 📚 Key Domains Covered
*   **Domain Concepts:** Defining what "unsafe" means for an embodied agent (physical constraints, object handling, etc.).
*   **Inference-Time Filtering:** Implementing fast, run-time checks on candidate actions before execution (e.g., VLESA). The foundational safety reports have been recently updated [here](VLA-guardrails/README.md) for the latest research details.
*   **Policy Alignment:** Training the core model to inherently prefer safe behavior through constrained optimization (e.g., SafeVLA).
*   **World Modeling:** Using predictive simulation to identify and avoid dangerous outcomes before they occur in reality (e.g., SafeDojo via [World Model Safety](openwiki/mechanisms/world_modeling_safety.md), or general prediction like World-Env).

---
*This documentation is continually updated as VLA safety research advances.*