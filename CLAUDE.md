# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project exploring **supervisor model distillation** — how a larger "supervisor" language model can distill its behavior into a smaller "worker" model. The project investigates multiple experimental settings:

- **Few-shot demonstrations**: Pointing to what information to relay in summaries
- **Many-shot demonstrations**: Gradient-based imitation learning
- **On-policy distillation**: Potentially after an SFT (supervised fine-tuning) stage, exploring large/small model collaboration

## Status

The project is in the early ideation phase. See `idea.md` for the research concept and related reading list. No code, build system, or dependencies have been set up yet.
