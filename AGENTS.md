# AGENTS.md

## Purpose
This repository is a Python research reimplementation project.
The goal is to convert legacy notebook-based research code into a clean, reproducible, modular codebase.

## Project context
This repository reimplements two internal research papers:
- "Deep Networks Learn Features from Local Discontinuities in the Label Function"
- "Layers as Lenses: A Narrative of Feature Learning in Deep Networks"

There are no reliable external references beyond:
- the papers stored in this repository,
- the old supplementary notebooks in `notebooks/old notebooks/`,
- notes written in this repository.

Treat these repository files as the source material.

## Source-of-truth policy
If information conflicts:
1. Prefer the paper over the notebook.
2. Prefer explicit experiment details over inferred behavior.
3. Do not silently guess missing details; flag ambiguities.

## Core development goal
First build a faithful, minimal, reproducible baseline implementation.
Only after that should the codebase be extended, generalized, or optimized.

## Repository structure policy
- `src/layer_lenses/` contains stable importable Python code.
- `configs/` contains experiment configuration files.
- `notes/` contains research specs, design notes, and experiment logs.
- `results/` contains metrics, plots, tables, checkpoints, and generated artifacts.
- `tests/` contains lightweight validation and regression tests.
- `notebooks/` contains exploratory work only.
- `notebooks/old notebooks/` contains legacy reference material only.

## Read-only paths
Treat the following as read-only unless the user explicitly asks otherwise:
- `notebooks/old notebooks/`
- legacy paper PDFs and supplementary files stored inside that directory

Do not edit, move, rename, or delete files in those paths.
Use them only for reading, summarizing, and extracting logic into clean modules.

## Notebook migration policy
Old notebooks are reference material, not the target implementation style.

When reading old notebooks:
- summarize the notebook purpose,
- identify data generation/loading logic,
- identify model logic,
- identify training loop logic,
- identify evaluation/plotting logic,
- identify duplicated or messy code,
- propose clean module destinations for stable logic.

Any logic required for reproducibility should be moved into `src/layer_lenses/` or `configs/`.
Do not leave essential experiment logic trapped in notebooks.

## Working style
When asked to do work in this repo:
1. Restate the task briefly.
2. Identify assumptions and relevant files.
3. Propose a small implementation plan.
4. Make minimal, reviewable changes.
5. Explain what changed and how to validate it.

## Session logging
At the end of any non-trivial session, append an entry to
`notes/session_log.md` following the format at the top of that file.

- Always preserve the user's prompts **verbatim** in the
  "User prompts (verbatim)" subsection. Do not paraphrase or summarize
  them.
- Do not store verbatim agent responses; summarize them in "Changes"
  and "Decisions" instead.
- If a decision is durable (paper vs notebook, a hyperparameter choice,
  a naming convention), also update the appropriate long-lived note
  (`paper_mismatches.md`, `reproducibility_log.md`, or this file) and
  link to it from the session entry.
- Trivial sessions (pure Q&A, no repo changes) can be skipped, but
  err on the side of logging.

## Coding preferences
- Prefer simple, explicit Python over clever abstractions.
- Use Python 3.12.
- Prefer small functions with clear names.
- Use type hints where helpful.
- Add short docstrings for public functions.
- Avoid unnecessary classes.
- Avoid adding dependencies unless clearly helpful.
- Preserve behavior when refactoring unless explicitly asked to change behavior.

## Reproducibility rules
- Put important settings in config files, not hardcoded constants.
- Expose seeds explicitly.
- Make training runnable from the command line.
- Save outputs and metadata under `results/`.
- Avoid hidden manual steps.
- Add lightweight tests for critical logic.

## First-phase priorities
Optimize for a clean first success:
- one baseline at a time,
- one clear config-driven run path,
- one reproducible evaluation path,
- one well-documented source of truth.

## When uncertain
If paper details, notebook behavior, and code structure disagree:
- stop and surface the mismatch,
- recommend options,
- ask for a decision instead of guessing.