# Paper vs Notebook Mismatches

Running log of places where the paper sources and the legacy notebooks disagree,
and how this repo resolves the disagreement. Each entry should cite the source
file(s), the observed difference, the chosen resolution, and the date noted.

Source-of-truth order (per `AGENTS.md`):
1. The paper.
2. Explicit experiment details in the paper.
3. Notebook behavior (only when the paper is silent or ambiguous).

## Open mismatches (to confirm while implementing milestone 1)

1. **Data sampling for `x` (single_path_dlgn vs multi_path_dlgn vs Paper 1 spec).**
   - `notebooks/old_notebooks/layers_as_lenses/code/single_path_dlgn.ipynb` (cell 8): draws `x` as `sign(standard_normal) @ rand_orth` (rotated hypercube vertices), and overwrites the decision vectors to axis-aligned `w_list[i,i]=1`.
   - `notebooks/old_notebooks/layers_as_lenses/code/multi_path_dlgn.ipynb` (cell 7) and Paper 1 (Sec. "Datasets used", "Synthetic Datasets"): `x` sphere-uniform, decision vectors unit-norm mutually orthogonal via QR.
   - **Resolution for milestone 1:** follow the paper — sphere-uniform `x`, QR-orthogonal `u_i`, bias 0.

2. **Gate activation in the single-path DLGN notebook.**
   - `single_path_dlgn.ipynb` uses a `ramp_sigmoid`.
   - Paper 1 Sec. 3.1 and Paper 2 Sec. 3.4 both specify `phi(u) = 1 / (1 + exp(-beta u))`.
   - **Resolution for milestone 1:** plain `sigmoid(beta*u)`. Ramp activation is not needed for milestone 1; if we later reproduce the exact single-path trajectory figure, we can add it as an opt-in activation.

3. **Label convention.**
   - Notebooks store labels as `{0, 1}` (via `vals = np.arange(...) % 2`).
   - Both papers use `y in {-1, +1}`.
   - **Resolution for milestone 1:** use `{-1, +1}` throughout. Sibling leaves get opposite signs (Paper 1, "Synthetic Datasets").

4. **Paper 2's "DLGN" vs Paper 1's taxonomy.**
   - Paper 2's model (Sec. 3.4) parameterizes gating directly by `w_{m,l}` vectors, i.e. by the effective weights `V^l`.
   - Paper 1 calls this parameterization **DLGN-SF** and reserves **DLGN** for the deep-linear-gating variant with parameters `W^1,...,W^L` and `V^l = W^l ... W^1`.
   - **Resolution:** In this repo, canonical names follow Paper 1. `DLGN_SF` is Paper 2's model. Milestone 1 uses `DLGN_SF`.

5. **Multi-path training hyperparameter: L2 gating penalty coefficient.**
   - Paper 2 Sec. 5.1 refers to "L2 penalty" without pinning the coefficient in the body. Figure folder names in the notebook (`Regularization_2e-4_phase1`) imply `lambda = 2e-4`.
   - **Resolution:** milestone 1 does not use any regularization. Two-phase / L2 will be revisited in phase 2.

## Notebook-specific gaps (noted, to be revisited)

- `notebooks/old_notebooks/layers_as_lenses/code/data/` and `models/` are empty — no shipped datasets or checkpoints; data and models must be regenerated.
- Paper 1 Table 1 says synthetic datasets SDI/II/III use total sample sizes 40k/60k/100k. Paper 1 Table 2's distance-table experiment uses "30000 data points" with `d=100, D=4, L=4, m=20`. Milestone 1 follows Table 2.

(New entries go at the top of the "Open mismatches" list.)
