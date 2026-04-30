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

---

# Extension Plan: Layers-as-Lenses Narrative for ReLU MLPs

Math-rendered copy: `notes/research_spec_math.ipynb` (use this in Cursor when equations do not render in `.md` preview).

This section captures the high-level research plan for extending the
"Layers as Lenses" narrative from DLGNs to standard fully connected ReLU MLPs
on COB-ODT data. It is *intentionally conceptual*: hypotheses, tests,
visualisations, and sequencing only — no code yet. It is the working
research charter for the ReLU analysis phase.

## A. Recap of the Paper's Core Narrative

The paper makes three durable claims that we want to extend to ReLU MLPs:

- **Lens–Blinder–Residue decomposition.** For DLGN, the gradient on neuron
$\mathbf{w}*k$ factors as Residue × Lens × Blinder × $y^* a \mathbf{x}$.
The Lens $\Lambda_k(\mathbf{x})=\prod*{\kappa\ne k}\phi(\mathbf{w}_\kappa^\top\mathbf{x})$ comes
from *other* neurons; the Blinder $\phi'(\mathbf{w}_k^\top\mathbf{x})$ comes from the
neuron itself.
- **Dual role of a neuron and positive feedback.** $\mathbf{w}_k$ is simultaneously
(a) a linear separator on data importance-weighted by its lens and
(b) a component of lens for everyone else. This dual role is the engine
of feature learning.
- **Hierarchy-seeking + self-opposing second-order dynamics.** These give
"leaf-first, then root, then descendants" feature emergence, but also
produce backlash along spurious directions, motivating the L2-then-prune
two-phase training in DLGN.

The narrative-level claim of the paper is broader than DLGN: this should
be a generic phenomenon for any deep non-linear architecture. The goal of
the ReLU analysis phase is to make this claim *legible* in a standard
ReLU MLP.

## B. Conceptual Translation: DLGN → ReLU MLP

This is the crux. We need clean ReLU analogs of three DLGN objects.

### B.1 Gates

- **DLGN-SF gate at layer $\ell$:** $\phi(\mathbf{w}_\ell^\top \mathbf{x})$ — a *shallow*
halfspace in input space (always reads $\mathbf{x}$ directly).
- **ReLU gate at layer $\ell$:** $\mathbb{1}[(W_\ell h_{\ell-1}(\mathbf{x}))*i > 0]$ — a
deep halfspace. For a fixed input $\mathbf{x}$ inside a single linear region,
it is still a halfspace in input space, but its normal depends on the
activation pattern at $\mathbf{x}$:
$$\tilde\mathbf{w}*{\ell,i}(\mathbf{x}) = \bigl((W_\ell)*{i,:}\text{diag}(\mathbf{m}*{\ell-1}(\mathbf{x}))W_{\ell-1}\cdots \text{diag}(\mathbf{m}_1(\mathbf{x}))W_1\bigr)^\top$$
  - For first-layer neurons, $\tilde\mathbf{w}*{1,i} = (W_1)*{i,:}$.
  - For deeper neurons, $\tilde\mathbf{w}_{\ell,i}(\mathbf{x})$ is *input-dependent*. The
  "gate" is not a single vector but a family of vectors over linear
  regions and must be aggregated sensibly (e.g., averaged over the
  neuron's active subset of inputs).

### B.2 Paths

- **DLGN path:** $\pi = (m_1,\ldots,m_L) \in [M]^L$ with multiplicative
gate product across layers.
- **ReLU path:** Same index $\pi$, but the path's contribution is
multilinear and gated:
$$f(\mathbf{x}) = \sum_{\pi} \Bigl[\prod_{\ell=1}^{L-1} m_{\ell,i_\ell}(\mathbf{x})\Bigr]\cdot a_\pi \cdot (W_1)*{i_1,:}^\top \mathbf{x}$$
with $a*\pi = w_{L,i_{L-1}}\prod_{\ell=2}^{L-1}(W_\ell)*{i*\ell,i_{\ell-1}}$.
So a ReLU path naturally has:
  - an **activation indicator** (subset of inputs that turn it on),
  - a **scalar coefficient** $a_\pi$,
  - a **linear separator** $(W_1)_{i_1,:}$ (shared across all paths
  sharing $i_1$).

### B.3 Lens–Blinder for ReLU

The ReLU gradient w.r.t. $(W_k)*{i,:}$ takes the same factorised form,
with the lens being a back-propagated sensitivity through the active
sub-network for $\mathbf{x}$:
$$\nabla*{(W_k)*{i,:}} \mathcal{L} = \mathbb{E}\bigl[\underbrace{\sigma(-y^*\hat y)}*{\text{Residue}}\cdot \underbrace{\Lambda^{\text{ReLU}}*{k,i}(\mathbf{x})}*{\text{Lens}}\cdot \underbrace{\mathbb{1}[(W_k h_{k-1})*i > 0]}*{\text{Blinder}}\cdot y^*h_{k-1}(\mathbf{x})^\top\bigr]$$
The lens for ReLU is a *path-summed* quantity: contributions from every
active path passing through neuron $(k,i)$. This is the natural ReLU
analog of $\Lambda_k(\mathbf{x})$ in DLGN.

## C. Hypotheses (numbered, each independently testable)

Grouped by the level of ambition.

### Tier A — Direct DLGN-style claims for first-layer ReLU

**H1 (First-layer alignment).** First-layer ReLU neurons in a successful
run align (in direction) with ODT decision vectors. Most large-norm
neurons have a high $|\cos|$ with some $\mathbf{u}_i$.

- *Test:* Track norm vs $|\cos|$ scatter over training; expect mass to
migrate from origin toward axis-aligned clusters at high $|\cos|$.

**H2 (Hierarchical emergence in first layer).** Neurons capture leaves
first, then root, then root's children, etc.

- *Test:* For each ODT level, plot fraction of first-layer neurons whose
argmax-$|\cos|$ lies at that level vs epoch. Expect leaf curves to rise
first, then deeper levels with delays.

**H3 (Order-of-emergence quantification).** Even within a level, more
"important" ODT directions (root before its siblings, etc.) are captured
earlier.

- *Test:* For each $\mathbf{u}_i$, define "capture epoch" as the first epoch any
neuron achieves $|\cos|\ge\tau$ with it. Plot capture epoch vs ODT
level / scope of discontinuity.

### Tier B — Path-level structure

**H4 (Path specialisation).** Over training, the active paths $\pi$ tend
to align with ODT root-to-leaf paths. The set of inputs activating a
given $\pi$ becomes (approximately) a subset of one ODT leaf.

- *Test:* For each path $\pi$, measure the entropy of its activation's
distribution over ODT leaves. Track this entropy over epochs; expect
it to drop. At convergence, top paths should map nearly 1-to-1 with
ODT leaves.

**H5 (Path coefficient sparsity / lottery-ticket parallel).** A small
number of paths carry most of the output magnitude.

- *Test:* Plot Lorenz curve / Gini of $|a_\pi|$ over training. Track
effective number of paths $\sim \exp(H(|a_\pi|/\sum|a_{\pi'}|))$.

**H6 (Path-level coverage of input space).** The union of top-$k$ paths'
activation regions covers nearly the entire input distribution and
aligns with ODT leaf partition.

- *Test:* Color test points by their dominant active path; compare
against true ODT leaf assignments via clustering ARI / NMI.

### Tier C — Lens–Blinder–Residue empirics

**H7 (Lens dominates direction near init).** Early in training, the
gradient direction for $\mathbf{w}_k$ is well-predicted by the lens-weighted
average of $y^*\mathbf{x}$. The blinder mostly modulates magnitude, not
direction.

- *Test:* For each first-layer neuron, decompose the empirical gradient
into the ratio of (lens × Residue × signed input) vs (blinder
modulation). Track this ratio over time.

**H8 (Lens reorientation).** As training progresses, the lens of each
neuron shifts from "diffuse" to "concentrated on one ODT leaf" — i.e.
each neuron's lens-weighted data subset becomes sharply localised.

- *Test:* For each neuron, compute the ODT-leaf distribution of its
lens-weighted (or for ReLU, gradient-weighted) training data. Track
entropy and dominant-leaf identity over time.

**H9 (Hierarchy-seeking analog).** Perturbing one neuron along a leaf
direction increases the gradient component of other neurons toward an
ancestor direction. (Direct ReLU analog of the paper's Theorem 1.)

- *Test:* Empirical second-order probe: pick a checkpoint, perturb
$W_1[i,:]\leftarrow W_1[i,:]+\epsilon\mathbf{u}_j$, recompute gradient on
another neuron, project on ancestor $\mathbf{u}_i$. Average over many input
batches. Compare against unperturbed gradient.

**H10 (Self-opposing analog).** Perturbing one neuron along a spurious
direction $\h \perp \mathbf{u}_i$ pushes other neurons along $-\h$ —
magnifying the spurious component for the population.

- *Test:* Same perturbation framework as H9 but with random spurious
$\h$. Aggregate second-order responses.

### Tier D — Dynamics, interventions, predictions

**H11 (Positive feedback ablation).** Freezing deeper layers (no lens
evolution) drastically slows or breaks first-layer ODT alignment.

- *Test:* Compare full training vs frozen-deep-layer training (only
first layer trains). Track first-layer alignment trajectories.

**H12 (Width and lottery tickets).** With wider first layer, the
fraction of neurons that ever align is roughly constant; absolute count
grows. Pruning to aligned neurons preserves accuracy.

- *Test:* Vary first-layer width. After training, prune all first-layer
neurons whose $|\cos|<\tau$ to all $\mathbf{u}_i$; retrain or fine-tune the
rest. Compare to original accuracy.

**H13 (Init-to-solution prediction).** A heuristic score (proximity +
hierarchy-seeking + self-opposition support) at initialization predicts
which ODT direction each first-layer neuron will eventually lock onto.

- *Test:* Direct port of the paper's Section 3.4 heuristic, adapted to
first-layer ReLU. Even if accuracy is modest, beating proximity-only
baseline is a strong signal.

### Tier E — Deeper layers (hardest)

**H14 (Effective gate alignment for deeper layers).** When you compute
effective gating vectors $\tilde\mathbf{w}_{\ell,i}(\mathbf{x})$ averaged across the
active subset of neuron $(\ell,i)$, they too align with ODT directions,
with later layers tending to pick *coarser* ODT levels (root/internal
nodes) rather than leaves.

- *Test:* For each $(\ell,i)$, compute mean effective gate over its
active inputs. Run the same scatter (norm vs $|\cos|$) and ODT-level
coloring. Compare across layers.

**H15 (Layer-as-lens specialisation).** Earlier layers act more as
"separators" (their gates are sharp halfspaces in $\mathbf{x}$); later layers
act more as "lens modulators" (they boost/suppress contributions from
earlier paths).

- *Test:* Quantify per-layer contributions to gradient lens vs separator
role; show monotone trend with depth.

## D. Visualisation Ideas (matched to hypotheses)

- **V1: First-layer norm vs $|\cos|$ scatter, level-coloured.** Already
in place. Add: animate over the quadratic snapshot epochs to see
emergence narrative; overlay the "capture epoch" of each ODT direction.
- **V2: Per-neuron trajectory traces.** For each first-layer neuron, a
curve $|\cos|$ vs epoch with a separate panel for norm vs epoch.
Optionally, mark which $\mathbf{u}_i$ is the argmax over time (and if it
switches). Helps test H1, H3, H13.
- **V3: ODT capture timeline (Gantt-style).** y-axis: ODT nodes ordered
by level; x-axis: epoch; bar starts when first neuron captures $\mathbf{u}_i$
above $\tau$. Helps test H2, H3.
- **V4: Path-level summaries.** For each epoch: path-frequency bar plot
(top 50 by activation count); per-path "ODT leaf entropy" over epochs.
Helps test H4, H5, H6.
- **V5: Lens-weighted data projection.** For a chosen neuron, compute
its lens (ReLU back-prop sensitivity through that neuron); project
lens-weighted training data onto $(\mathbf{u}_i, \mathbf{u}_j)$ planes for various
$(i,j)$. Track how the projection concentrates over time. Helps test
H7, H8.
- **V6: Effective gate scatter for deeper layers.** For each $(\ell,i)$,
compute mean effective gate $\bar{\tilde\mathbf{w}}_{\ell,i}$. Plot the same
norm vs $|\cos|$ scatter, layer by layer. Helps test H14, H15.
- **V7: Second-order probe heatmap.** Matrix indexed by (perturbation
direction $\h \in \mathbf{u}_j$, measurement direction $\z\in\mathbf{u}_i$);
entry = empirical second-order push (averaged over neurons). Compare
to the theoretical sign pattern from Theorems 1–2. Helps test H9, H10.
- **V8: Lottery-ticket pruning curves.** After training, sweep a
norm/cos threshold and prune first-layer neurons. Plot test accuracy
vs pruning fraction. Helps test H12.
- **V9: Init-to-solution prediction confusion matrix.** For multiple
seeds: predicted vs actual eventual ODT lock-in for each first-layer
neuron. Helps test H13.

## E. Suggested Sequencing

Given that we currently have one successful single-seed ReLU run, the
next conversations should sequence like this (each can be a small
focused next step):

1. **Solidify first-layer narrative (H1–H3) with V1–V3.** Make sure the
  "leaves-then-root-then-children" story holds. Closest port of the
   paper's Figs 2/3 to ReLU and confirms that the framework transfers.
2. **Path-level objects (H4–H6) with V4 + V8.** Define and instrument
  paths as first-class entities. Conceptually new step beyond DLGN
   tooling.
3. **Lens–Blinder empirics (H7–H8) with V5.** Pick one or two
  illustrative neurons; show the lens reorientation story.
4. **Second-order probes (H9–H10) with V7.** Direct empirical validation
  of the paper's theorems in the ReLU setting — very compelling if it
   works.
5. **Interventions (H11–H13).** Freezing, pruning, init-prediction —
  these test the explanatory/predictive power of the framework, which
   is what the paper emphasises ("explain, predict, understand").
6. **Deeper layers (H14–H15).** The hardest. Best to attempt only after
  the first five steps stabilise the narrative.

## F. Pitfalls to Watch

- **Linear-region explosion:** ReLU has $O(2^{LM})$ linear regions in
principle. We can't enumerate per region; aggregate quantities
(per-neuron averages over its active set, per-path means over active
inputs) are the right granularity.
- **Sign ambiguity:** ReLU is not sign-symmetric like the gating in
DLGN; so $|\cos|$ may need to be replaced by signed cosine, or
analyzed with care.
- **Bias terms:** Current `ReLUMLP` uses `bias=True`. This shifts gate
boundaries off the origin. The COB-ODT data is on the unit sphere
with zero bias, so to get clean alignment we may want to also study
the `bias=False` variant.
- **Permutation invariance:** Like DLGN, ReLU layers are
permutation-invariant. Per-neuron tracking must allow re-identification
across epochs (Hungarian matching by $|\cos|$, or by parameter
trajectory continuity).
- **Identifiability of paths:** Many paths share the same $i_1$ and may
be degenerate. Path-level analysis should focus on "effectively
distinct" paths via clustering on $a_\pi$ and activation-region.
- **Successful vs unsuccessful runs:** Some seeds may fail (saddle
points). Studying both, and what distinguishes them, is itself
informative — this is what the paper does in Section 3.4.

## G. Open Decisions for Next Conversations

- Whether to formally define ReLU paths as first-class objects in the
codebase, and how to compute them efficiently.
- What subset of H1–H6 to prioritise for a first deep dive.
- Whether to add a `bias=False` parallel run for cleanly comparing
first-layer alignment.

