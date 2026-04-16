# Research Spec

## Project goal

This repository is a clean reimplementation of two related research projects from our group:

1. "Deep Networks Learn Features from Local Discontinuities in the Label Function"
2. "Layers as Lenses: A Narrative of Feature Learning in Deep Networks"

The goal is to convert messy supplementary notebook code into a clear, reproducible, modular Python project. The first objective is faithful reimplementation, not novelty. Extensions should come only after the baseline pipeline is correct and reproducible.

## Source of truth

There are no meaningful external web references for this project.
The source of truth is:

- the attached papers,
- the old supplementary notebooks in `notebooks/old notebooks/`,
- any notes added in this repository.

If there is a conflict:

1. trust the paper over the notebook,
2. trust explicit experiment details over implied behavior,
3. document ambiguities instead of guessing silently.

## First milestone

Build a minimal clean implementation that can:

- generate or load the required data setting,
- instantiate the core model cleanly,
- run training from a config file,
- evaluate and save outputs to `results/`,
- reproduce one baseline result from each paper or a clearly defined subset.

## Non-goals for the first phase

- No premature abstraction for many experiment types.
- No large framework before one successful baseline run.
- No extensions until baseline behavior is documented and reproducible.
- No notebook-only logic that is required for rerunning experiments.

## Code organization principles

- `src/first_experiment/` contains stable importable code.
- `notebooks/` is for exploration and temporary analysis only.
- `notes/` contains paper notes, experiment specs, and design decisions.
- `configs/` contains experiment configs.
- `results/` contains metrics, plots, tables, checkpoints, and artifacts.

## Notebook migration rule

Old notebooks are reference material, not final implementation.
Any logic needed for reproducibility should be migrated into clean Python modules.
Useful notebook content should be extracted into:

- data generation/loading code,
- model definitions,
- training/evaluation utilities,
- analysis utilities.

## Implementation priorities

1. Understand and summarize the old notebooks.
2. Identify reusable components and duplicated logic.
3. Build a minimal baseline implementation for one paper first.
4. Add tests and config-driven runs.
5. Only then add extensions or generalization.

## Open questions

- Which parts of the notebook code are essential versus exploratory?
- What exact experiments are required for a faithful baseline?
- Which hyperparameters and data settings are truly part of the claimed result?

