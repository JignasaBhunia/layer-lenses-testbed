# Session Log

Append-only log of agent working sessions. Newest entries go at the **bottom**.

## Purpose

A fresh agent (or a future you) should be able to read the most recent
few entries and know:
- what was being worked on,
- what changed in the repo and why,
- what's still open, and
- the exact framing the user used (via preserved prompts).

This file complements, not replaces:
- `AGENTS.md` — stable project conventions.
- `notes/research_spec.md` — project goals and invariants.
- `notes/paper_mismatches.md` — durable paper-vs-notebook decisions.
- `notes/reproducibility_log.md` — hyperparameter provenance per config.

Code, tests, and configs remain the source of truth for *what currently
runs*. This file captures the *narrative and intent* around changes.

## Entry format

```
## YYYY-MM-DD HH:MM IST — <short session title>

### Goal
One or two lines on what the session set out to do.

### User prompts (verbatim)
- "..."
- "..."

### Changes
- Files touched and a brief note on each.

### Decisions
- Non-obvious decisions, with links to paper_mismatches.md or
  reproducibility_log.md if applicable.

### Current state / where to pick up
- What's working, what's half-done, what's next.

### Open questions
- Anything that needs the user's input or further investigation.
```

Keep entries short. Use IST for all timestamps. If exact historical
times are unknown, keep the date and mark the heading as `(IST)` without
an HH:MM time. Always append new entries at the end of this file only
(never insert between existing entries). If a decision is durable
(paper vs notebook, a
hyperparameter choice, a naming convention), migrate it into the
appropriate long-lived note and link to it from here.

---

## 2026-04-22 (IST) — Retroactive snapshot of repo state

This entry predates the session-log policy and is written from a
cold read of the repo (git log, source files, notes, tests). It is a
best-effort reconstruction, not a transcript. Correct as needed.

### Current state / where to pick up

**Milestone 1 target:** reproduce the shared "DLGN gating hyperplanes
cluster around COB-ODT decision hyperplanes" phenomenon from
Paper 1 (Table 2 setting) and Paper 2 (Fig 2 style scatter).
See `notes/reproducibility_log.md` §`configs/milestone1_dlgn_sf_odt.yaml (planned)`.

**Implemented in `src/first_experiment/`:**
- `odt.py` — COB-ODT synthetic data generation. Unit-sphere `x`,
  QR-orthonormal decision normals, zero bias, deterministic leaf-sign
  assignment by index parity with optional global flip. Includes
  `samples_reaching_node` path utility.
- `dlgn.py` — `DLGNSF` (Paper 2's DLGN / Paper 1's DLGN-SF). Gating
  `sigmoid(beta * V_l x)`, `value_input_mode in {"ones", "x"}`, same
  value-network parameterization in both modes, configurable
  gating/value init scales. Final logit is multiplied by 2 to mimic the
  legacy notebook's `[-z, z]` + cross-entropy behavior under BCE.
- `training.py` — `TrainConfig` (Adam, BCE-with-logits, lr=2e-3,
  batch_size=1024, epochs=200, seed=365 by default), `train_dlgn_sf`
  with per-epoch shuffling, `evaluate_dlgn_sf` (log loss, 0/1 loss,
  accuracy), full-state-dict checkpoint snapshots at
  `snapshot_epochs`, and `effective_gating_weights_from_checkpoint`.
- `parity_debug.py` — utilities for comparing current code against
  the legacy notebook. Used to confirm parity.

**Tests (`tests/`):**
- `test_dlgn_training.py` — forward-shape sanity, parameter-count
  parity across value-input modes, init-scale math, smoke train-loss
  decrease, eval metric shape.
- `test_odt.py` — COB-ODT data-generation checks.

**Notebooks:**
- `notebooks/legacy_dlgn_minimal.ipynb` — minimal legacy reproduction
  (reference).
- `notebooks/scratch.ipynb` — current exploratory workspace (has
  uncommitted changes at the time of this snapshot).
- `notebooks/old_notebooks/` — read-only legacy reference per `AGENTS.md`.

**Configs:** `configs/` is empty. The milestone-1 YAML described in
`reproducibility_log.md` is planned but not yet written.

**Recent git history (most recent first):**
- `f837f83` — testing legacy vs current code variation across seeds (2).
- `23ef54c` — testing legacy vs current code variation across seeds.
- `af0d01e` — reproduced DLGNSF comparable with legacy notebook:
  depth-5 ODT, 80k samples, ~10% error in phase 1.
- `9867ea5` — scaffolding updates.
- `637ad6a` — initial DLGN model + training; scratch notebook testing
  ongoing.
- `3213db1` — ODT path utilities (`samples_reaching_node`).
- `6e5745c` — ODT data-gen code added.
- `3d6b6b6` — milestone-1 scaffolding and reproducibility notes.

### Open questions / known gaps

- No `configs/milestone1_dlgn_sf_odt.yaml` yet; training is currently
  driven from notebooks rather than from a committed config file.
  This is the natural next milestone-1 step.
- No CLI entry point (`AGENTS.md` calls for config-driven runs from the
  command line).
- No analysis-output code yet for `figures/scatter_init.pdf`,
  `figures/scatter_final.pdf`, or the Paper 1 Table 2–style
  `distance_table.csv`.
- `notebooks/scratch.ipynb` and `src/first_experiment/dlgn.py` have
  uncommitted changes as of this snapshot; their intent is not captured
  in git history yet.

### User prompts (verbatim)

None captured for sessions predating this policy. Cursor's local
transcripts in `~/.cursor/projects/.../agent-transcripts/` remain the
fallback for older sessions.

---

## 2026-04-22 (IST) — Establish session-log policy

### Goal
Set up a durable, in-repo context-preservation mechanism so that a new
agent can come up to speed after a Cursor reinstall or agent
termination.

### User prompts (verbatim)
- "This project was built with a previous Cursor agent's help. But an update crashed it and I had to reinstall cursor. I suppose that would leave you with little context on the state of the project. You can look at the markdown notes in the notes folder to get a sense of the project. You can read any other files in the project to build your context.\n\nTo prevent something like this happening again, can you create a conversations.md file in the notes folder with the chats I have with you and a brief summary of your text response (if not the full response)? Is that a good way to maintain the context with AI coding agents, that can be used for getting a new agent upto speed if the old agent has been terminated due to some issue? Or is there a better way? The reprodicibility_log notebook already serves a similar purpose."
- "Ok session_log.md is fine. But do maintain the exact text of my inputs in the chat, even if you don't store your chat outputs, which could be a lot of noise. My text inputs should not be very large even for a long project, simply by human limitations. Any existing note or a new note can be used for this log."

### Changes
- Created `notes/session_log.md` (this file) with format, policy, a
  retroactive state snapshot of the repo as of 2026-04-22, and this
  entry.
- Updated `AGENTS.md` with a working-style rule: agents must append a
  session-log entry (including verbatim user prompts) at the end of any
  non-trivial session.

### Decisions
- **Do not** keep a verbatim log of agent responses. Chat transcripts
  are low-signal-per-token and Cursor already stores them locally.
- **Do** keep verbatim user prompts. They are bounded in volume by
  human input and they preserve framing and intent — the most useful
  signal for a cold-start agent.
- Combine narrative state and verbatim prompts in a single file
  (`session_log.md`) rather than splitting across two files; split
  later if it grows unwieldy.

### Current state / where to pick up
- Policy is set up. Next time a non-trivial code change is made, append
  an entry at the bottom of this file following the format above.
- Milestone-1 work itself is unchanged; see the retroactive snapshot
  above for where that stands.

### Open questions
- None for this meta-session.

---

## 2026-04-22 (IST) — Fix legacy notebook log loss

### Goal
Make minimal edits to the legacy notebook so loss is computed directly
as binary cross-entropy on scalar logits, instead of first converting
to 2-class logits `[-y, y]` for cross-entropy.

### User prompts (verbatim)
- "Can you make minimal changes to the @notebooks/legacy_dlgn_minimal.ipynb  notebook that only fixes the log loss computation? Instead of making an output y into a [-y, y] before computing cross-entropy loss, make a direct binary cross-entropy loss from the model ouptut y. Change all places that use the cross-entropy loss with this modification."

### Changes
- Updated notebook cell 4 (`give_loss`) to use
  `nn.BCEWithLogitsLoss()` directly on `preds`, with float targets
  reshaped to `(N, 1)`.
- Updated notebook cell 13 (`train_dlgn`) to replace all
  `nn.CrossEntropyLoss()` usage and all `torch.cat((-1*..., ...), dim=1)`
  constructions with direct `nn.BCEWithLogitsLoss()` on scalar logits.
- Kept edits minimal and localized to the loss computation path only.

### Decisions
- Targets are cast to `float32` and reshaped to match model logit shape
  for BCE-with-logits compatibility.
- No other notebook logic (model architecture, data generation,
  evaluation/error plotting) was changed.

### Current state / where to pick up
- `notebooks/legacy_dlgn_minimal.ipynb` now has no remaining
  `CrossEntropyLoss` usage and no `[-y, y]` logits construction in the
  training/loss code paths.

### Open questions
- None.

---

## 2026-04-30 19:36 IST — Add neuron-to-leaf activity-vector helper

### Goal
Add a reusable helper in `relu_analysis.py` that converts `neuron_leaf_activity_summary` into a dictionary mapping each neuron to a leaf-ordered activity vector.

### User prompts (verbatim)
- "Can you write a piece of code that takes neuron_leaf_activity_summary as input and creates dictionary with each key corresponding to a neuron, and value corresponding to a numpy array of length equal to the number of leaf nodes in the ODT? Each element in this array should correspond to the total points in the leaf that are active for that neuron."
- "PLease do so, and put it in the @src/first_experiment/relu_analysis.py file."

### Changes
- Updated `src/first_experiment/relu_analysis.py` with:
  - `neuron_leaf_activity_dict(neuron_leaf_activity_summary, *, key_mode='global')`
    returning `(mapping, leaf_order)`.
  - Supports `key_mode='global'` and `key_mode='layer_local'`.
  - Validates required input columns and handles empty input safely.

### Decisions
- Return both mapping and `leaf_order` to make the vector index-to-leaf correspondence explicit and robust in downstream analysis.

### Current state / where to pick up
- Use:
  - `mapping, leaf_order = neuron_leaf_activity_dict(neuron_leaf_activity_summary)`
  - or `key_mode='layer_local'` for `(layer, local_neuron_id)` keys.
- Lints are clean and import check via `uv run python` succeeded.

### Open questions
- None.

---

## 2026-04-30 19:11 IST — Add neuron activity count by ODT leaf

### Goal
Add a reusable helper for counting neuron activity within a specified ODT leaf and use it in the scratch notebook to build an all-neuron/all-leaf activity summary.

### User prompts (verbatim)
- "Write a function which takes a model checkpoint, a neuron id, and a ODT label leafnode id, and gives the total number of points with that label leaf node id for which that neuron is active. Have this function be in relu_analysis.py

Use this function in scratch.ipynb to get an appropriate summary object for all neurons and ODT nodes. Insert this cell just before the start/end metrics cell."

### Changes
- Updated `src/first_experiment/relu_analysis.py`:
  - Added `count_active_points_for_neuron_leaf(...)` to count active points for a given hidden neuron and leaf node.
  - Added supporting internal helpers for precomputing hidden activation masks / leaf IDs and resolving neuron IDs.
  - Added `summarize_neuron_leaf_activity(...)` convenience helper.
- Updated `notebooks/scratch.ipynb`:
  - Inserted a new code cell immediately before the `## Start/end metrics` section that loops all hidden neurons and ODT leaf nodes and builds `neuron_leaf_activity_summary` using `count_active_points_for_neuron_leaf`.
  - Left two previously inserted duplicate cells as no-op comments indicating that the summary moved.

### Decisions
- Accept both global neuron ids (`0..119`) and `(layer_idx, local_neuron_idx)` tuples in the counting helper for flexibility.
- Use leaf IDs in ODT heap indexing (`31..62` for depth 5), matching current notebook conventions.

### Current state / where to pick up
- Run the new pre-start-metrics cell to populate `neuron_leaf_activity_summary`.
- Use the resulting dataframe directly or with `itables.show(...)` for interactive filtering.

### Open questions
- None.

---

## 2026-04-30 16:53 IST — Extract ReLU notebook analysis helpers

### Goal
Move reusable functions out of `notebooks/scratch.ipynb` into importable source code while keeping plots and exploratory analysis in the notebook.

### User prompts (verbatim)
- "I want the plots and analyses to be on the scratch notebook, but would like to move all functions to an appropriate (maybe new or one of the old ones) .py file. \n\nIf any other piece of code in the notebook (not yet as a function) can also be moved into a function that would also be good.\n\nCan you do this reorganisation? "

### Changes
- Added `src/first_experiment/relu_analysis.py` with reusable ReLU/COB-ODT helpers:
  - single-seed ReLU run setup,
  - start/end metric collection,
  - checkpoint model loading,
  - first-layer ODT alignment computation and scatter plotting,
  - layered ReLU graph plotting,
  - masked ReLU forward/eval by ODT leaf,
  - single-neuron ablation responsibility analysis,
  - compact responsibility summary generation.
- Updated `notebooks/scratch.ipynb` to import those helpers and replaced notebook function definitions with calls.
- Replaced inline scatter/graph mechanics with calls to `plot_first_layer_odt_alignment` and `plot_layered_relu_graph`.

### Decisions
- Created a new `relu_analysis.py` rather than extending `analysis.py`, because existing `analysis.py` is focused on multipath DLGN aggregate statistics.
- Kept plotting calls in the notebook, but moved plotting implementations into source so the notebook remains a thin analysis driver.

### Current state / where to pick up
- Re-run the first import/config cell in `notebooks/scratch.ipynb` before running downstream analysis cells.
- `relu_analysis.py` imported successfully through `uv run python`; IDE lints report no errors for the touched notebook and module.

### Open questions
- None.

---

## 2026-04-30 09:29 IST — Add ReLU neuron leaf-responsibility ablation

### Goal
Add notebook tooling to assign hidden ReLU neurons to ODT label leaves by measuring per-leaf log-loss increase after single-neuron ablation.

### User prompts (verbatim)
- "In the cell before the start metrics, I am trying to see if the neurons (all the 120 neurons) can be assigned to one or multiple ODT nodes -- i.e. I want to say \"this neuron is responsible for getting data entering these ODT label leaf nodes classified correctly\".  One way to do this is as follows: have a mask that is all ones, and zero only that neuron and look at all ODT nodes where the log loss is much higher than if we had all neurons active. \n\nCan you write a function which takes in a model and outputs a dictionary with keys being (layer, neuron_id) and values capturing the ODT nodes that neuron is responsible for. Ideally it would also capture the impact of that neuron e.g. how much does the log loss go up by for those ODT nodes etc. \n\nInsert this function and a small script calling it appropriately before the start/end metrics cell."

### Changes
- Inserted a new code cell in `notebooks/scratch.ipynb` before the start/end metrics section.
- Added `neuron_leaf_responsibility_from_ablation`, which compares all-active per-leaf metrics against one-neuron-ablated metrics for all hidden neurons.
- Added an example call that produces `neuron_responsibility`, `neuron_responsibility_df`, `baseline_leaf_metrics`, and a compact `neuron_summary` display.

### Decisions
- Use per-leaf `mean_log_loss` deltas as the primary responsibility signal, with optional thresholds for relative log-loss delta and accuracy drop.
- Keep layer indices zero-based to match `hidden_layers` indexing in `ReLUMLP`.

### Current state / where to pick up
- Run the new notebook cell after the per-leaf masked evaluation helper has been defined and after `seed_result`, `ckpts`, and `EPOCH` exist.
- Tune `min_log_loss_delta` to control how many ODT leaves are attributed to each neuron.

### Open questions
- None.

---

## 2026-04-30 08:52 IST — Add per-leaf masked ReLU log loss

### Goal
Extend the masked ReLU ODT-leaf evaluation helper in the scratch notebook to report average log loss per leaf, not only accuracy.

### User prompts (verbatim)
- "Can you make a change directly in the notebook for also returning the average log loss in each leaf node?"

### Changes
- Updated `notebooks/scratch.ipynb` helper `masked_accuracy_by_odt_leaf` to compute per-example BCE/logistic loss from `{−1,+1}` labels and logits using `np.logaddexp`.
- Added `mean_log_loss` to the returned per-leaf summary table.

### Decisions
- Use the numerically stable `{−1,+1}` logistic-loss form `logaddexp(0, -y * logit)` to match the BCE-with-logits objective used during training.

### Current state / where to pick up
- Re-run the helper-definition cell and then the leaf-accuracy display cell to see the new `mean_log_loss` column.

### Open questions
- None.

---

## 2026-04-25 08:43 IST — Simplify scratch notebook to single-seed focus

### Goal
Remove multi-seed distractions from `scratch.ipynb` and focus all
learning-dynamics analysis on one selected seed.

### User prompts (verbatim)
- "I want to analyse the learning dynamics for a single seed now. Having multiple seeds in all results in other places distracts me. Can you make this modification to the scratch.ipynb notebook?"

### Changes
- Updated `notebooks/scratch.ipynb` to single-seed workflow:
  - Replaced `MASTER_SEEDS` + `all_results` with:
    - `SEED = 5178`
    - `seed_result = None`
  - Replaced training loop over seeds with one run:
    - `seed_result = run_single_seed(SEED)`.
  - Updated scatter cell to read from `seed_result` and use `SEED` in
    title; added guard if training hasn’t run yet.
  - Replaced aggregate multi-seed table cell with single-seed start/end
    table (`metrics_table`).
  - Updated final visualization cell to plot from `metrics_table`
    instead of `agg_table`; added guard if training hasn’t run yet.
  - Updated markdown headings/descriptions to reflect single-seed focus.

### Decisions
- Kept helper function `collect_start_end_metrics(...)` unchanged,
  because it already supports a single seed result input cleanly.

### Current state / where to pick up
- Notebook is now organized around one seed and no longer presents
  multi-seed aggregation outputs in the main flow.
- You can switch seed by changing `SEED` in one place.

### Open questions
- None.

---

## 2026-04-24 19:22 IST — Switch to dense-early quadratic snapshots

### Goal
Capture early-training transitions better by replacing uniform
`SNAPSHOT_STRIDE=200` with a dense-near-init snapshot schedule.

### User prompts (verbatim)
- "Most interesting transitions in the neural network happen early in training. Having snapshots only every 200 epochs does not capture it. Can we capture snapshots more frequently close to init? e.g. quadratically spaced snapshots? 0,1,2,4,7,11,16,22,29, etc?"

### Changes
- Updated `notebooks/scratch.ipynb` config cell:
  - Removed fixed `SNAPSHOT_STRIDE`-based snapshot generation.
  - Added `quadratic_snapshot_epochs(total_epochs, n_points=80)`.
  - Set `SNAPSHOT_EPOCHS` from quadratic spacing over `[0, TOTAL_EPOCHS]`.
  - Added printouts for count + early/late snapshot values for quick
    verification.

### Decisions
- Used quadratic spacing (`frac**2`) to make snapshots much denser near
  initialization and gradually sparser later in training.

### Current state / where to pick up
- Training now stores more early checkpoints while limiting total
  snapshot count.
- Scatter/analysis cells continue to work with `SNAPSHOT_EPOCHS`.

### Open questions
- None.

---

## 2026-04-24 19:17 IST — Add first-layer ReLU vs ODT scatter cell

### Goal
Add a scatter-plot analysis cell in `scratch.ipynb` (before aggregate
analysis) showing first-layer ReLU weight norms against max absolute
cosine similarity to ODT vectors, colored by the best-matching ODT node.

### User prompts (verbatim)
- "Create the code for the scatter plot cell in scratch.ipynb where the weight vector norm of the first layer relu net is plotted w.r.t the max abs cosine similarity to ODT vectors. Color the points based on the ODT node maximising this abs cosine similarity. Similar to the scatter plots for the DLGN made earlier.

Make this in the cell before \"aggregate analysis\"."

### Changes
- Updated `notebooks/scratch.ipynb` cell before “Aggregate analysis”:
  - Validates chosen checkpoint epoch exists.
  - Loads first-layer ReLU weights from checkpoint
    (`hidden_layers.0.weight`).
  - Computes per-neuron weight norm.
  - Computes cosine similarities with ODT normals (`tree.w_list`),
    then max absolute cosine and argmax ODT node index.
  - Renders scatter:
    - x: first-layer weight norm
    - y: max absolute cosine similarity to ODT vectors
    - color: ODT node index maximizing abs cosine
  - Adds colorbar, labels, title, and light grid.

### Decisions
- Implemented with torch ops throughout for consistency with checkpoint
  tensors and numerical stability (`clamp_min(1e-12)` in denominator).

### Current state / where to pick up
- Notebook now has DLGN-style alignment scatter for the ReLU first layer
  in the requested position.

### Open questions
- None.

---

## 2026-04-24 15:34 IST — Add progress option to ReLU training

### Goal
Add an option in ReLU training to show epoch-wise progress.

### User prompts (verbatim)
- "Can you add an option in the relu training for showing progress?"

### Changes
- Updated `src/first_experiment/relu_training.py`:
  - Added `show_progress: bool = False` to `ReLUTrainConfig`.
  - Added tqdm support in `train_relu_mlp(...)`:
    - wraps epoch iteration with a progress bar when enabled
    - shows current epoch loss via `set_postfix`.

### Decisions
- Mirrored DLGN training progress behavior for consistency across
  training modules.

### Current state / where to pick up
- ReLU training now supports progress display with
  `ReLUTrainConfig(show_progress=True)`.
- Validation passed:
  - `./.venv/bin/python -m pytest` -> `24 passed`.

### Open questions
- None.

---

## 2026-04-24 15:30 IST — Move ReLU training routine into Python module

### Goal
Move ReLU training logic out of `scratch.ipynb` into a `.py` training
module while keeping experiment orchestration/data generation in the
notebook.

### User prompts (verbatim)
- "It is better to have the training routine in a .py file . But there is already a training.py file that is for the DLGN, maybe you can reuse the same file or create another file and rename accordingly."

### Changes
- Added `src/first_experiment/relu_training.py`:
  - `ReLUTrainConfig`
  - `train_relu_mlp(...)`
  - `evaluate_relu_mlp(...)`
  - internal PM1-to-binary label conversion helper
- Updated `notebooks/scratch.ipynb`:
  - Replaced notebook-local ReLU train/eval implementations with imports
    from `first_experiment.relu_training`.
  - Kept data generation and seed orchestration in notebook code.
  - `run_single_seed(...)` now calls `train_relu_mlp(...)` with
    `ReLUTrainConfig`.
- Added `tests/test_relu_training.py` for new module coverage.

### Decisions
- Chose a separate `relu_training.py` instead of merging into DLGN
  `training.py` to keep model-specific training paths clean and avoid
  coupling.

### Current state / where to pick up
- ReLU training routine is now reusable and importable from Python code.
- Notebook remains experiment-driver for data generation and analysis.
- Validation passed:
  - `./.venv/bin/python -m pytest` -> `24 passed`.

### Open questions
- None.

---

## 2026-04-24 15:26 IST — Simplify to single-phase ReLU workflow in notebook

### Goal
Reorganize ReLU experiment code to remove multi-phase analysis utilities,
keep only `relu_mlp.py`, and move experiment/training logic into
`scratch.ipynb`.

### User prompts (verbatim)
- "Reorganise the code. There is no need yet for multi-phase analysis. Only a single phase of training is required for now. Keep only the relu_mlp file, and move all experimental details in relu_experiments.py to scratch.ipynb. Have the data generation and training happen in scratch itself using calls to odt.py etc. Remove the relu_experiments.py file."

### Changes
- Deleted `src/first_experiment/relu_experiments.py`.
- Deleted `tests/test_relu_experiments.py` (depended on removed module).
- Updated `notebooks/scratch.ipynb` to a single-phase-only ReLU pipeline:
  - Replaced multi-phase content with direct notebook-local helpers.
  - Imports now use `odt.py`, `training.set_seed`, and `relu_mlp.py`.
  - Added notebook-local training loop (`train_relu_single_phase`) and
    evaluation helper (`evaluate_relu_mlp`).
  - Added notebook-local `run_single_seed` that generates data and trains
    one-phase model per seed.
  - Updated aggregate analysis and plots to compare start vs end
    metrics only (no run-mode axis).

### Decisions
- Kept only `relu_mlp.py` as reusable architecture code, per request.
- Moved all experiment orchestration details (configs, training loop,
  evaluation flow) into `scratch.ipynb`.

### Current state / where to pick up
- ReLU workflow is now single-phase and notebook-driven.
- Test suite passes after cleanup:
  - `./.venv/bin/python -m pytest` -> `22 passed`.

### Open questions
- None.

---

## 2026-04-24 15:18 IST — Add standard ReLU MLP experiment pipeline

### Goal
Create a standard fully connected ReLU architecture path (no dropout,
no normalization), add required experiment files, and repopulate
`scratch.ipynb` for analysis on this architecture.

### User prompts (verbatim)
- "Thanks. Now I need to perform these experiments with a different architecture. Simply the standard fully connected ReLU is sufficient for now. Can you create the required files, and populate the (now empty) scratch.ipynb for performing analysis with it? Choose appropriate hyperparameters as configs (number of hidden layers, number of neurons per layer etc). No need for extra layer magics like dropout or normalization for now."

### Changes
- Added `src/first_experiment/relu_mlp.py`:
  - Introduced `ReLUMLP`, a plain fully connected ReLU classifier with
    scalar-logit output.
- Added `src/first_experiment/relu_experiments.py`:
  - Added `ReLUTrainConfig` and `ReLUExperimentConfig`.
  - Added `evaluate_binary_classifier(...)`.
  - Added `train_relu_mlp(...)` with Adam + cosine schedule + optional
    row-freeze for pruned hidden neurons.
  - Added `run_relu_single_seed(...)` implementing:
    - `two_phase_phase1`
    - `two_phase_phase2_no_prune`
    - `two_phase_phase2_pruned`
    - `one_phase`
- Repopulated `notebooks/scratch.ipynb`:
  - Added config/imports cells for ReLU experiments.
  - Added helper cell to collect start/end checkpoint metrics.
  - Added full multi-seed run cell.
  - Added aggregate per-seed and mean/std analysis tables.
  - Added quick end-metric visualization plots.
- Added `tests/test_relu_experiments.py`:
  - Forward-shape sanity for `ReLUMLP`.
  - Training/evaluation smoke coverage.
  - Single-seed run-mode output coverage.

### Decisions
- Used `hidden_dims=(256, 256, 256)` as a strong but still standard
  baseline for this first ReLU pass.
- Kept training loop simple (no dropout/norm/other layer magic) per
  request.
- Preserved previous experiment framing (one-phase vs two-phase with and
  without pruning) while swapping the architecture.

### Current state / where to pick up
- ReLU experiment path is now available in package code and notebook.
- `scratch.ipynb` is analysis-ready for multi-seed ReLU runs.
- Validation passed:
  - `./.venv/bin/python -m pytest` -> `25 passed`.

### Open questions
- None.

---

## 2026-04-24 10:53 IST — Stop storing split arrays in `all_results`

### Goal
Reduce serialized notebook cache size by removing `x_train`, `x_eval`,
`y_train`, `y_eval` from `all_results` and regenerating splits from
`data_seed` when needed for analysis.

### User prompts (verbatim)
- "Can we avoid storing the x_train, x_eval, y_train, y_eval as part of all_results? We could just generate those from the seeds, right?"
- "Yes, please do so."

### Changes
- Updated `notebooks/multiseed_multipath_runner.ipynb`:
  - Added helper function:
    `regenerate_split_from_seed(data_seed)` to deterministically
    reconstruct train/eval splits from generation settings.
  - Updated `run_single_seed_with_checkpoints(...)` return payload to
    remove the `"data"` block (`x_train`, `y_train`, `x_eval`, `y_eval`)
    from `all_results`.
  - Updated start/end aggregate metrics cell to regenerate split arrays
    from `seed_result["seeds"]["data_seed"]` instead of reading stored
    arrays.

### Decisions
- Kept deterministic regeneration path in-notebook because dataset
  generation is seed-driven and split slicing is fixed.
- Chose to reduce cache payload size rather than storing duplicated
  arrays per seed.

### Current state / where to pick up
- `all_results` is now lighter and still supports scatter and aggregate
  analysis.
- Saved pickle caches will no longer carry train/eval arrays if produced
  after re-running the updated notebook cells.

### Open questions
- None.

---

## 2026-04-23 22:26 IST — Remove redundant re-training API block from notebook

### Goal
Confirm whether the notebook `run_multiseed(...)` section retrains models
and remove it if redundant relative to already-trained checkpoint-based
analysis flow.

### User prompts (verbatim)
- "Does the cell using the run_multiseed API (## Aggregate stats path using the new API (`run_multiseed`)) in the multiseed_multipath_runner.ipynb file train the model again?  Training run over all the run modes takes about 40 minutes per seed.

Does this cell (## Aggregate stats path using the new API (`run_multiseed`)) give me any more functionality than the cells before it and after it does? If not and it actually trains again please remove it."

### Changes
- Updated `notebooks/multiseed_multipath_runner.ipynb`:
  - Removed imports used only by the `run_multiseed(...)` section:
    `ExperimentConfig`, `DataConfig`, `ModelConfig`, `run_multiseed`.
  - Replaced the markdown section text to explicitly mark the
    `run_multiseed` path as removed.
  - Replaced the `run_multiseed(...)` code cell with a placeholder
    comment to prevent accidental full retraining.

### Decisions
- Confirmed `run_multiseed(...)` does retrain all modes for all seeds.
- Since equivalent start/end aggregates are already computed from
  `all_results` (existing checkpoint-rich run), keeping the API block in
  this notebook was redundant and expensive.

### Current state / where to pick up
- Notebook now performs only one training pass (`all_results`) and uses
  those outputs for scatter plots and aggregate start/end tables.
- Accidental duplicate retraining from the removed section is prevented.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add init-time freezing for zeroed gating rows

### Goal
Add init-time parameter freezing support in training, then use it in the
scratch notebook's zeroed Phase-2 branch so zeroed gating vectors remain
frozen during Phase 2.

### User prompts (verbatim)
- "Need functionality in the training routine to freeze some parameters at init. This is to be used in the two phase training second phase to freeze the gating vectors set to zero. Make changes to both the training sub-routine and the scratch notebook to reflect this change."

### Changes
- Updated `src/first_experiment/training.py`:
  - Added `TrainConfig.freeze_parameter_names` for exact-name parameter
    freezing at initialization.
  - Added `TrainConfig.freeze_zero_gating_rows` to freeze all-zero
    gating rows detected at init.
  - Added validation for unknown entries in `freeze_parameter_names`
    (raises `ValueError`).
  - Implemented row-level gradient masking hooks for zeroed gating rows.
  - Added post-step projection to keep frozen gating rows exactly zero.
  - Expanded train function docstring to document freezing controls.
- Updated `tests/test_dlgn_training.py`:
  - Added `test_train_dlgn_sf_freeze_zero_gating_rows_keeps_zeroed_rows_fixed`.
  - Added `test_train_dlgn_sf_unknown_frozen_parameter_raises`.
  - Ran tests: `./.venv/bin/python -m pytest tests/test_dlgn_training.py -q`
    passed (`10 passed`).
- Updated `notebooks/scratch.ipynb`:
  - In the zeroed Phase-2 branch, introduced
    `train_cfg_phase2_zeroed` with `freeze_zero_gating_rows=True`.
  - Wired `out_two_phase_phase2_zeroed` to use this config.
  - Updated run selector map so
    `two_phase_phase2_zeroed` uses `train_cfg_phase2_zeroed`.
  - Updated selector title text to indicate rows are both pre-zeroed and
    frozen.

### Decisions
- Kept freezing behavior opt-in and backward compatible (defaults remain
  non-freezing).
- Used row-level freezing for the gating weights to match the notebook
  use case without requiring architecture changes.

### Current state / where to pick up
- Two-phase branch now supports:
  - Phase 2 without zeroing (`out_two_phase_phase2_nozero`)
  - Phase 2 with pre-zeroed + frozen low-norm gating rows
    (`out_two_phase_phase2_zeroed`)
- Training routine supports both generic named-parameter freezing and
  automatic freezing of zero-initialized gating rows.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add alternate zeroed-gating Phase 2 branch

### Goal
Extend the two-phase notebook pipeline so Phase 1 is shared, then run
two different Phase-2 branches: with and without pre-zeroing low-norm
gating vectors, alongside the one-phase baseline.

### User prompts (verbatim)
- "Build an alternate Phase 2 for the two phase training. The main difference is that before the second phase begins, all gating neurons whose weight vector norm is less than a threshold is zeroed. It also uses the same Phase 1.\n\nSo now there are 4 training runs that has to happen:\n\nPhase 1\nPhase 2 after zeroing some gating vectors\nPhase 2 without any zeroing of gate vectors\nSingle phase."

### Changes
- Updated `notebooks/scratch.ipynb` two-phase section:
  - Added `GATING_NORM_ZERO_THRESHOLD = 0.2`.
  - Kept the same Phase-1 output (`out_two_phase_phase1`) as the shared
    starting point.
  - Added Phase-2 branch A:
    `out_two_phase_phase2_nozero` (no pre-zeroing).
  - Added Phase-2 branch B:
    `out_two_phase_phase2_zeroed` after setting to zero all gating-layer
    rows with norm below threshold.
  - Kept compatibility alias:
    `out_two_phase_phase2 = out_two_phase_phase2_nozero`.
- Updated variable-reference markdown cell to document:
  - `out_two_phase_phase2_nozero`
  - `out_two_phase_phase2_zeroed`
  - `GATING_NORM_ZERO_THRESHOLD`
- Updated metrics run selector cell:
  - Added `RUN_MODE` options for both Phase-2 variants:
    `two_phase_phase2_nozero` and `two_phase_phase2_zeroed`.

### Decisions
- Used `copy.deepcopy(...)` to ensure both Phase-2 branches start from
  the same Phase-1 state without mutating each other.
- Performed gating-row zeroing under `torch.no_grad()` so it is a direct
  parameter edit before Phase-2 training begins.

### Current state / where to pick up
- Notebook now supports four runs:
  - `out_two_phase_phase1`
  - `out_two_phase_phase2_nozero`
  - `out_two_phase_phase2_zeroed`
  - `out_one_phase`

### Open questions
- None.

---

## 2026-04-22 (IST) — Correct IST format and chronology (no exact times)

### Goal
Switch session-log headings to IST conventions, correct chronology from
context, and leave historical times blank where exact HH:MM is unknown.

### User prompts (verbatim)
- "Use IST for the time stamps. Some of the time stamps are clearly wrong and end up giving the updates in the wrong order. (e.g. LR scheduling log is the latest entry prior to this meta conversation) If you don't have the exact times the previous log entries were made, please re-order them correctly from context and leave the times for those blank."

### Changes
- Updated heading guidance to IST usage with explicit fallback:
  date + `(IST)` when exact HH:MM is unknown.
- Converted existing same-day headings to date-only `(IST)` style where
  earlier times were uncertain.
- Reordered entries so chronology follows context:
  pre-metrics cleanup → variable reference → metrics selector →
  LR scheduler → append/timestamp meta fix.

### Decisions
- Avoided fabricating exact historical times; used date-only `(IST)` for
  prior entries that could not be timed reliably.

### Current state / where to pick up
- Session log now reads in correct chronological order and uses IST
  convention without guessed timestamps.

### Open questions
- None.

---

## 2026-04-22 (IST) — Scratch notebook pre-metrics cleanup

### Goal
Apply minimal fixes in `notebooks/scratch.ipynb` before the "Metrics at
each snapshot epoch" section: reproducible seeding (including model
init), clearer two-phase/one-phase naming, and consistent checkpoint
creation setup.

### User prompts (verbatim)
- "I have built a pipeline in scratch.ipynb to compare the two-phase training (gating regularisation followed by no-regularisation with an optional not-yet implemented pruning step in-between) to the no-regularisation one phase only training. In scratch.ipynb. Can you go over it and see if it serves the intended purpose? You can suggest minor renaming/consistency edits. \n\nOne other issue I noticed is reproducibility with random seeds. Randomness is used in three places -- data-generation, model initialisation and training mini-batch selection. Only data-gen and SGD randomness are seeded. The init is not. Can you suggest fixes for that too?"
- "Yes, make these minimal edits up to the training/creation of these checkpoint snapshots for both the workflows. I have not yet started to go through the plotting/comparisons yet. So only fix the issues pointed out by you in the cells before the \"Metrics at each snapshot epoch\" markdown cell."

### Changes
- Updated imports/config cell to include `set_seed` and explicit seed
  variables: `DATA_SEED`, `INIT_SEED`, `TRAIN_SEED_PHASE1`,
  `TRAIN_SEED_PHASE2`, `TRAIN_SEED_ONE_PHASE`.
- Switched data generation to use `seed=DATA_SEED`.
- Added explicit initialization seeding before two-phase model creation
  (`set_seed(INIT_SEED)`), and renamed model variable to
  `model_two_phase`.
- Updated the two-phase training cell to:
  - use fair epoch accounting (`TOTAL_EPOCHS`, split into
    `PHASE1_EPOCHS` and `PHASE2_EPOCHS`),
  - use renamed run outputs (`out_two_phase_phase1`,
    `out_two_phase_phase2`),
  - use separate training seeds for phase 1 and phase 2 configs.
- Renamed one markdown header to "one-phase no-regularization run".
- Updated the one-phase training cell to:
  - reseed initialization before creating `model_one_phase`,
  - use `train_cfg_one_phase` and `out_one_phase`,
  - keep temporary aliases (`out`, `train_cfg`, `out_phase2`) for
    downstream compatibility with existing metric/plot cells.

### Decisions
- Kept edits strictly before the metrics markdown section as requested.
- Preserved backward-compatible aliases to avoid breaking later cells
  before a dedicated plotting/comparison cleanup pass.

### Current state / where to pick up
- Both workflows now have explicit checkpoint-producing training paths
  with controlled data/init/train randomness.
- The metrics and plotting sections remain intentionally untouched and
  still reference compatibility aliases.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add scratch variable reference cell

### Goal
Add a short markdown reference cell before the metrics section in
`notebooks/scratch.ipynb` listing analysis-relevant variable names and
their meanings.

### User prompts (verbatim)
- "Can you create a markdown cell before the metric plotting cell to give the variable names (out_one_phase, train_cfg_one_phase etc) that are relevant for analysis later with a brief description?"

### Changes
- Inserted a new markdown cell immediately before the metrics markdown
  section in `notebooks/scratch.ipynb`.
- The cell documents:
  - `out_one_phase`, `train_cfg_one_phase`
  - `out_two_phase_phase1`, `train_cfg_phase1`
  - `out_two_phase_phase2`, `train_cfg_phase2`
  - compatibility aliases (`out`, `train_cfg`, `out_phase2`)

### Decisions
- Kept the text concise and analysis-focused, without modifying any
  metric/plot code.

### Current state / where to pick up
- Notebook now includes an explicit pre-metrics variable reference for
  downstream analysis.

### Open questions
- None.

---

## 2026-04-22 (IST) — Metrics run selector in scratch notebook

### Goal
Replace manual metrics-cell assignment with a small selectable block for
choosing one-phase or either two-phase stage, including automatic
`fig.suptitle`.

### User prompts (verbatim)
- "Can you make a small piece of code that assigns out, train_cfg, and sets fig.suptitle based on one of three options: one phase, two phase-phase1 or two-phase-phase2 ? Something that can be pasted at the top of the metrics cell to replace the current manual assignment?"
- "Please do."

### Changes
- Updated the metrics plotting code cell in `notebooks/scratch.ipynb`
  to add:
  - `RUN_MODE` selector with options:
    `\"one_phase\"`, `\"two_phase_phase1\"`, `\"two_phase_phase2\"`
  - `run_map` dictionary mapping each mode to:
    `out`, `train_cfg`, and title text
  - mode validation (`ValueError` for invalid mode)
  - dynamic assignment:
    `out = run_map[RUN_MODE][\"out\"]`,
    `train_cfg = run_map[RUN_MODE][\"train_cfg\"]`,
    `FIG_SUPTITLE = run_map[RUN_MODE][\"title\"]`
  - dynamic suptitle:
    `fig.suptitle(FIG_SUPTITLE)`

### Decisions
- Kept all downstream metric/evaluation logic untouched; only the manual
  top-of-cell selection block was replaced.

### Current state / where to pick up
- Metrics cell now supports one-line workflow switching via `RUN_MODE`
  without editing variable names manually.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add LR scheduler to training routine

### Goal
Add a learning-rate scheduler to the core training routine with a
reasonable decay behavior and test coverage.

### User prompts (verbatim)
- "In the training routine, can you add a learning rate scheduler that decays appropriately?"

### Changes
- Updated `src/first_experiment/training.py`:
  - Added `TrainConfig.lr_scheduler` with options `\"none\"` (default)
    and `\"cosine\"`.
  - Added `TrainConfig.lr_scheduler_eta_min_ratio` to control cosine
    minimum LR (`eta_min = lr * ratio`).
  - Added scheduler setup in `train_dlgn_sf` and per-epoch
    `scheduler.step()` call after each training epoch.
  - Added `lr_history` to the returned training output dictionary.
  - Added validation error for unsupported scheduler names.
- Updated `tests/test_dlgn_training.py`:
  - Added `test_train_dlgn_sf_cosine_scheduler_decays_learning_rate`
    to verify LR decreases across epochs when cosine scheduling is
    enabled.
- Ran tests:
  - `./.venv/bin/python -m pytest tests/test_dlgn_training.py -q`
    passed (`8 passed`).

### Decisions
- Kept behavior backward-compatible by defaulting scheduler to
  `\"none\"`; existing experiments keep fixed LR unless explicitly
  configured.
- Used cosine annealing as the built-in decay strategy because it
  decays smoothly and needs only epoch count + minimum LR ratio.

### Current state / where to pick up
- Training now supports both fixed-LR and cosine-decay modes through
  `TrainConfig`.
- Existing notebook/config code can opt in with
  `lr_scheduler=\"cosine\"`.

### Open questions
- None.

---

## 2026-04-22 (IST) — Fix append order and add timestamps in session log

### Goal
Fix session-log ordering issues (entries being inserted in the middle)
and include a time stamp in heading dates.

### User prompts (verbatim)
- "The session logs are being inserted in the middle and not appended. Can you also add a time stamp to the date there?"

### Changes
- Updated session-log heading format template from
  `YYYY-MM-DD` to `YYYY-MM-DD HH:MM`.
- Added an explicit instruction in this file: always append new entries
  at the end, never insert between existing entries.
- Updated existing 2026-04-22 entry headings to include time stamps.
- Reordered misplaced recent entries so newest entries appear lower in
  the file (append-style progression).

### Decisions
- Used timestamped headings consistently to avoid ambiguity and make
  ordering easier to audit.

### Current state / where to pick up
- `notes/session_log.md` now has timestamped headings and explicit
  append-only guidance.

### Open questions
- None.

---

## 2026-04-23 21:53 IST — Refactor multipath workflow into scriptable APIs

### Goal
Implement the approved refactor plan to move multipath DLGN experiment
logic from notebook orchestration into stable Python modules, including
multi-seed execution, descriptive statistics, artifact IO, and tests.

### User prompts (verbatim)
- "Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos."
- "log the changes in session_log."

### Changes
- Added `src/first_experiment/experiment_config.py`:
  - Introduced experiment-level contracts:
    `SeedOffsets`, `SeedBundle`, `DataConfig`, `ModelConfig`,
    `ExperimentConfig`, `EvalMetrics`, `PruningSummary`,
    `RegimeResult`, `SeedRunResult`.
  - Encoded branch-specific `TrainConfig` builders for phase-1,
    phase-2 (no-zero), phase-2 (zeroed/frozen), and one-phase runs.
- Added `src/first_experiment/experiment_runner.py`:
  - Implemented `run_one_phase`, `run_two_phase_nozero`,
    `run_two_phase_zeroed`, and `run_all_regimes_for_seed`.
  - Ported notebook branch logic (shared phase-1 base model, optional
    pre-phase-2 zeroing by norm threshold, freeze-compatible training).
  - Added pruning invariants (`rows_zeroed_per_layer`,
    `rows_still_zero_after_training`).
- Added `src/first_experiment/analysis.py`:
  - Implemented descriptive aggregation across seeds:
    `mean`, `std`, `n`, `sem`, `ci95_low`, `ci95_high`.
  - Added baseline-delta summary helper
    `compute_delta_vs_baseline(...)`.
- Added `src/first_experiment/io.py`:
  - Added run-root creation, manifest writing, per-seed JSON output,
    and summary CSV writing under `results/`.
- Added `src/first_experiment/multiseed.py`:
  - Implemented API-first multi-seed orchestration (`run_multiseed`)
    with optional artifact persistence.
- Added tests:
  - `tests/test_experiment_runner.py`
  - `tests/test_multiseed_analysis.py`
  - Coverage includes seed-fairness invariants, zeroed-branch freeze
    invariants, descriptive schema checks, and artifact writes.
- Validation run:
  - `./.venv/bin/python -m pytest tests/test_experiment_runner.py tests/test_multiseed_analysis.py tests/test_dlgn_training.py`
  - Result: `15 passed`.

### Decisions
- Kept implementation API-first (as requested) while preserving a
  deterministic artifact structure for later CLI wiring.
- Scoped statistics to descriptive summaries + 95% CI for this
  milestone; deferred inferential tests/effect sizes.
- Used normal-approximation 95% CI in `analysis.py` for initial
  reviewer-facing summaries; this can be replaced with t-based CI in a
  later refinement pass if required.

### Current state / where to pick up
- Multipath orchestration is now available from importable Python APIs
  in `src/first_experiment/`.
- Multi-seed runs and summary table generation are available via
  `run_multiseed(...)`, with optional writing to `results/`.
- New tests for runner/analysis contracts pass alongside existing
  training tests.
- Next practical step: adapt `notebooks/scratch.ipynb` to consume these
  APIs and read saved outputs instead of owning execution logic.

### Open questions
- Whether to switch CI computation from normal-approximation to t-based
  intervals for small sample counts.
- Whether to add hypothesis testing/effect sizes in the next milestone
  for reviewer rebuttal material.

---

## 2026-04-23 22:05 IST — Create fresh multi-seed notebook template

### Goal
Provide a copy-paste-ready fresh notebook for running 5-seed multipath
experiments with lower checkpoint frequency, scatter support per
seed/mode/checkpoint, and aggregate statistics.

### User prompts (verbatim)
- "Let's say my goal is to run the experiments in scratch.ipynb for 5 different seeds. Can you give me code that can be copy pasted into a fresh notebook? Intersperse python cells with markdown cells describing the code cells below.

Potential issues include running out of memory becuase we store checkpoints every 20 epochs in all training runs. Maybe reduce that to every 200 epochs for convenience. I need to have the capability to get the scatter plots of the checkpoint epochs of any of the 5 seeds, and all the run modes using that seed config. I also need aggregate statistics such as training loss, test error at the beginning and end of all the run modes for all the chosen seeds.

Other settings such as epoch numbers, depth of tree, number of data points can be taken from the curent scratch.ipynb."
- "Please do so. While you are at it, just create the notebook also and put these cells in it."

### Changes
- Added new notebook:
  `notebooks/multiseed_multipath_runner.ipynb`.
- Populated notebook with interleaved markdown/python cells covering:
  - 5-seed multipath training workflow.
  - Scratch-aligned defaults (`DIM`, `DEPTH`, `N_TRAIN`,
    `TOTAL_EPOCHS`, hidden dims, `BETA`).
  - Reduced checkpoint cadence (`SNAPSHOT_STRIDE=200`).
  - Per-seed checkpoint-rich execution for scatter plots.
  - Scatter helper for any seed/mode/checkpoint:
    `plot_scatter_for_checkpoint(...)`.
  - Aggregate summaries via new API modules:
    `ExperimentConfig` + `run_multiseed(...)` for descriptive/final
    summaries and baseline deltas.
  - Explicit start/end checkpoint metrics aggregation
    (`train_log_loss`, `test_error`) across all seeds and modes.

### Decisions
- Kept two complementary paths in the notebook:
  - direct checkpoint-heavy run path for scatter flexibility;
  - `run_multiseed(...)` API path for script-aligned aggregate outputs.
- Retained scratch defaults while only lowering snapshot frequency to
  reduce memory pressure.

### Current state / where to pick up
- Notebook is ready to run end-to-end.
- User can change `MASTER_SEEDS` and `SNAPSHOT_STRIDE` in one place.
- Scatter and aggregate tables are both wired.

### Open questions
- None.

---

## 2026-04-25 10:13 IST — ReLU layers-as-lenses extension plan

### Goal
Draft a high-level research plan to extend the "Layers as Lenses"
narrative from DLGN to standard ReLU MLPs on COB-ODT data (gates,
paths, neuron-level dynamics) and persist it in a referenceable note.

### User prompts (verbatim)
- "My goal in this research project is to explain, predict and understand where and how neural network parameters move during training. The layers as lenses paper (one of the two papers in research_spec.md and in the old_notebooks folder) tries to do this for a simple single path DLGN and extends it to multi-path DLGNs. \n\nMy goal now is to do similar analysis for ReLU networks and maybe extend it further. I need to extend the notion of gates in DLGN (which are simply halfspaces in the input space) to ReLU networks. And paths in DLGNs (which are simply a product of gates) also to ReLU networks. These are no longer as simple in ReLU MLPs. But the phenomena they cause are similar. I believe the learning dynamics (even in ReLU MLPs) can be explained using the concept of a positive feedback between the dual roles of a neuron. \n\nTo that end I have taken a single successful ReLU run on a ODT dataset for analysis in scratch.ipynb. I am interested in studyng neuron level evolution of gates, as well as the evolution of paths. \n\nDo read the layers as lenses paper again (available both as pdf and tex file in the project) and the scratch.ipynb notebook and its results and think of a high level plan of action for hypotheses and ways of testing/visualising these. No need for code suggestions right now. Take your time and thnk deeply. We will have more conversations and brainstorming of ideas once you are done."
- "Great, this sounds like a good starting point. Move this to research_spec.md so that I can refer to it easier in future conversations."

### Changes
- Re-read `notebooks/old_notebooks/layers_as_lenses/Layers_as_Lenses.tex`
  and `notebooks/scratch.ipynb` to anchor the plan in current state.
- Appended a new section to `notes/research_spec.md`:
  "Extension Plan: Layers-as-Lenses Narrative for ReLU MLPs". Covers
  recap of paper claims, conceptual translation of gates/paths/lens to
  ReLU, 15 numbered hypotheses (Tiers A–E), 9 visualisation ideas,
  suggested sequencing, pitfalls, and open decisions.

### Decisions
- Plan kept conceptual only (no code changes) per user's explicit
  instruction.
- Hypotheses organised into tiers by ambition so future sessions can
  pick a subset to operationalise without re-deriving the framing.

### Current state / where to pick up
- Research charter for the ReLU analysis phase lives in
  `notes/research_spec.md` under the "Extension Plan" section.
- `notebooks/scratch.ipynb` still holds the single-seed ReLU run that
  the plan refers to (one successful seed, snapshot-rich training,
  level-coloured first-layer scatter already implemented).
- No code/test changes were made this session.

### Open questions
- Which of H1–H6 to prioritise for the first deep dive.
- Whether to introduce a `bias=False` parallel ReLU run for cleaner
  first-layer alignment comparisons.
- Whether to define ReLU "paths" as first-class objects in
  `src/first_experiment/` or keep them as analysis-time constructs
  inside the notebook for now.

---

## 2026-04-25 10:21 IST — Fix Markdown LaTeX rendering in research spec

### Goal
Make the math in `notes/research_spec.md` render correctly in Cursor's
Markdown viewer.

### User prompts (verbatim)
- "Latex is not rendered properly in the markdown viewer on cursor. Can you check why? e.g. in researchspec.md."
- "Please do so."

### Changes
- Updated LaTeX expressions in `notes/research_spec.md` to avoid custom
  paper macros unsupported by Markdown renderers.
- Replaced shorthand macros with standard forms used by KaTeX/MathJax:
  `\w, \x, \bu, \m, \E, \1` -> `\mathbf{w}, \mathbf{x}, \mathbf{u},
  \mathbf{m}, \mathbb{E}, \mathbb{1}`.
- Fixed one intermediate replacement artifact (`\mathbf{m}ath...`) and
  verified no unsupported shorthand macros remain.

### Decisions
- Keep equations renderer-friendly in notes by avoiding dependency on
  `.tex` preamble macros.

### Current state / where to pick up
- The math in `notes/research_spec.md` now uses standard Markdown-safe
  LaTeX and should render in Cursor.

### Open questions
- None.

---

## 2026-04-25 10:24 IST — Diagnose markdown LaTeX rendering

### Goal
Identify why LaTeX was not rendering in Cursor markdown preview and apply a fix.

### User prompts (verbatim)
- "No I still don't see proper rendering. Can you create a small markdown test file with a simple latex command?"
- "No this also does not render. How to fix this?"

### Changes
- Added `notes/latex_render_test.md` with minimal inline and block math checks.
- Updated Cursor user settings at
  `~/Library/Application Support/Cursor/User/settings.json` to include:
  - `"markdown.math.enabled": true`

### Decisions
- Treat issue as renderer/settings-level (not file content syntax), because even minimal test math did not render.

### Current state / where to pick up
- Project markdown files contain valid KaTeX-style math delimiters (`$...$`, `$$...$$`).
- Cursor now has explicit markdown math enabled in user settings.
- If still not visible, next check is whether user is viewing source editor vs markdown preview pane and reload window.

### Open questions
- None.

---

## 2026-04-25 10:39 IST — Add notebook copy for math-rendered research plan

### Goal
Provide a renderable (equation-friendly) version of the ReLU extension plan,
since Cursor markdown preview does not render LaTeX reliably in `.md` files.

### User prompts (verbatim)
- "It still fails. Maybe relevant: Markdown cells in .ipynb files seem to render fine."
- "Yes please do so."

### Changes
- Added `notes/research_spec_math.ipynb` containing the ReLU extension plan
  as markdown content with equations for notebook rendering.
- Updated `notes/research_spec.md` to add a pointer line under the
  "Extension Plan" heading:
  - Math-rendered copy: `notes/research_spec_math.ipynb`.

### Decisions
- Keep `notes/research_spec.md` as the canonical plain-text spec and use
  the notebook as the render-focused companion for math-heavy reading.

### Current state / where to pick up
- Use `notes/research_spec_math.ipynb` for rendered equations in Cursor.
- Use `notes/research_spec.md` for text-first reference and search.

### Open questions
- None.
