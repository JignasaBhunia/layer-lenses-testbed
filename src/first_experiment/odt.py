"""COB-ODT synthetic data generation utilities.

This module implements the Paper 1 synthetic dataset defaults used in milestone 1:
- inputs sampled uniformly from the unit sphere,
- complete orthogonal balanced ODT internal hyperplanes,
- zero biases by default,
- labels in {-1, +1} with sibling leaves assigned opposite signs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class COBODTSpec:
    """Configuration for a complete orthogonal balanced ODT."""

    dim: int
    depth: int
    threshold: float = 0.0

    @property
    def num_internal_nodes(self) -> int:
        return (2**self.depth) - 1

    @property
    def num_leaf_nodes(self) -> int:
        return 2**self.depth


@dataclass(frozen=True)
class COBODTTree:
    """Tree parameters defining the ODT label function."""

    w_list: np.ndarray
    b_list: np.ndarray
    leaf_labels: np.ndarray

    @property
    def depth(self) -> int:
        """Depth inferred from internal-node count for complete trees."""
        num_internal = self.w_list.shape[0]
        depth_float = np.log2(num_internal + 1)
        depth = int(depth_float)
        if (2**depth) - 1 != num_internal:
            raise ValueError(
                "w_list does not correspond to a complete binary tree. "
                f"Got {num_internal} internal nodes."
            )
        return depth


def _sample_unit_sphere(rng: np.random.Generator, num_data: int, dim: int) -> np.ndarray:
    x = rng.standard_normal((num_data, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _sample_orthonormal_vectors(
    rng: np.random.Generator, *, num_vectors: int, dim: int
) -> np.ndarray:
    if num_vectors > dim:
        raise ValueError(
            "Cannot sample mutually orthonormal vectors when num_vectors > dim. "
            f"Got num_vectors={num_vectors}, dim={dim}."
        )
    gaussian = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(gaussian)
    return q[:, :num_vectors].T


def _default_leaf_labels(depth: int, global_sign: int = 1) -> np.ndarray:
    if global_sign not in (-1, 1):
        raise ValueError(f"global_sign must be -1 or +1, got {global_sign}.")
    num_leaf_nodes = 2**depth
    parity = np.arange(num_leaf_nodes) % 2
    labels = np.where(parity == 0, -1, 1)
    return labels.astype(np.int8) * np.int8(global_sign)


def _traverse_tree(margins: np.ndarray, depth: int) -> np.ndarray:
    curr_index = np.zeros(margins.shape[0], dtype=np.int64)
    for _ in range(depth):
        decision = margins[np.arange(margins.shape[0]), curr_index]
        went_right = (decision > 0).astype(np.int64)
        curr_index = (2 * curr_index) + 1 + went_right
    return curr_index


def _validate_tree(spec: COBODTSpec, tree: COBODTTree) -> None:
    expected_internal = spec.num_internal_nodes
    if tree.w_list.shape != (expected_internal, spec.dim):
        raise ValueError(
            "w_list has wrong shape. "
            f"Expected {(expected_internal, spec.dim)}, got {tree.w_list.shape}."
        )
    if tree.b_list.shape != (expected_internal,):
        raise ValueError(
            "b_list has wrong shape. "
            f"Expected {(expected_internal,)}, got {tree.b_list.shape}."
        )
    if tree.leaf_labels.shape != (spec.num_leaf_nodes,):
        raise ValueError(
            "leaf_labels has wrong shape. "
            f"Expected {(spec.num_leaf_nodes,)}, got {tree.leaf_labels.shape}."
        )
    if not np.all(np.isin(tree.leaf_labels, (-1, 1))):
        raise ValueError("leaf_labels must only contain -1 and +1.")


def _path_directions_from_root(node_id: int) -> list[int]:
    """Return root-to-node directions (0=left, 1=right)."""
    directions: list[int] = []
    curr = node_id
    while curr > 0:
        parent = (curr - 1) // 2
        is_right = 1 if curr == (2 * parent + 2) else 0
        directions.append(is_right)
        curr = parent
    directions.reverse()
    return directions


def _infer_depth_from_tree(tree: COBODTTree) -> int:
    """Infer tree depth from number of internal nodes."""
    return tree.depth


def samples_reaching_node(
    *,
    x: np.ndarray,
    tree: COBODTTree,
    node_id: int,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return all samples that reach a given internal or leaf node.

    Args:
        x: Input samples of shape (n, d).
        tree: COB-ODT parameters.
        node_id: Node index in heap order (root=0).
        return_mask: If True, also return a boolean mask over `x`.
    """
    depth = _infer_depth_from_tree(tree)
    num_internal = (2**depth) - 1
    num_total_nodes = (2 ** (depth + 1)) - 1
    if node_id < 0 or node_id >= num_total_nodes:
        raise ValueError(
            f"node_id must be in [0, {num_total_nodes - 1}], got {node_id}."
        )

    if x.ndim != 2:
        raise ValueError(f"x must be 2D with shape (n, d), got shape {x.shape}.")
    if tree.w_list.shape[1] != x.shape[1]:
        raise ValueError(
            "x dimension does not match tree hyperplanes: "
            f"x has d={x.shape[1]}, tree expects d={tree.w_list.shape[1]}."
        )
    if tree.w_list.shape[0] != num_internal:
        raise ValueError(
            "tree does not match provided depth: "
            f"expected {num_internal} internal nodes, got {tree.w_list.shape[0]}."
        )

    directions = _path_directions_from_root(node_id)
    mask = np.ones(x.shape[0], dtype=bool)
    curr_node = 0
    for go_right in directions:
        margins = x[mask] @ tree.w_list[curr_node] + tree.b_list[curr_node]
        next_mask = margins > 0 if go_right == 1 else margins <= 0
        full_next_mask = np.zeros_like(mask)
        full_next_mask[np.where(mask)[0][next_mask]] = True
        mask = full_next_mask
        curr_node = 2 * curr_node + 1 + go_right

    x_subset = x[mask]
    if return_mask:
        return x_subset, mask
    return x_subset


def build_default_cob_odt_tree(
    *, dim: int, depth: int, rng: np.random.Generator, global_leaf_sign: int = 1
) -> COBODTTree:
    """Create default COB-ODT parameters following Paper 1 defaults."""

    num_internal_nodes = (2**depth) - 1
    w_list = _sample_orthonormal_vectors(rng, num_vectors=num_internal_nodes, dim=dim)
    b_list = np.zeros(num_internal_nodes, dtype=np.float64)
    leaf_labels = _default_leaf_labels(depth=depth, global_sign=global_leaf_sign)
    return COBODTTree(w_list=w_list, b_list=b_list, leaf_labels=leaf_labels)


def generate_cob_odt_data(
    *,
    num_data: int,
    dim: int,
    depth: int,
    seed: int,
    threshold: float = 0.0,
    tree: COBODTTree | None = None,
) -> tuple[np.ndarray, np.ndarray, COBODTTree, dict[str, int | float]]:
    """Generate synthetic data from a complete orthogonal balanced ODT.

    Returns:
        x_pruned: Input vectors on the unit sphere after threshold pruning.
        y_pruned: Labels in {-1, +1}.
        tree: ODT parameters used for generation.
        meta: Counts useful for reproducibility and diagnostics.
    """

    spec = COBODTSpec(dim=dim, depth=depth, threshold=threshold)
    rng = np.random.default_rng(seed)

    if tree is None:
        tree = build_default_cob_odt_tree(dim=dim, depth=depth, rng=rng)
    _validate_tree(spec, tree)

    x = _sample_unit_sphere(rng, num_data=num_data, dim=dim)
    margins = x @ tree.w_list.T + tree.b_list

    leaf_indices = _traverse_tree(margins, depth=depth)
    leaf_offsets = leaf_indices - spec.num_internal_nodes
    y = tree.leaf_labels[leaf_offsets]

    min_abs_margin = np.min(np.abs(margins), axis=1)
    keep = min_abs_margin > threshold

    x_pruned = x[keep]
    y_pruned = y[keep].astype(np.int8)

    meta: dict[str, int | float] = {
        "num_requested": int(num_data),
        "num_kept": int(x_pruned.shape[0]),
        "num_pruned": int(num_data - x_pruned.shape[0]),
        "dim": int(dim),
        "depth": int(depth),
        "num_internal_nodes": int(spec.num_internal_nodes),
        "num_leaf_nodes": int(spec.num_leaf_nodes),
        "threshold": float(threshold),
        "seed": int(seed),
    }
    return x_pruned, y_pruned, tree, meta
