"""Analysis helpers for ReLU MLPs trained on COB-ODT data.

These utilities keep exploratory notebooks focused on plotting choices and
interpretation while preserving the reusable mechanics in importable code.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from layer_lenses.odt import COBODTTree, generate_cob_odt_data, odt_leaf_ids_for_x
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import (
    ReLUTrainConfig,
    checkpoint_model_from_state,
    evaluate_relu_mlp,
    train_relu_mlp,
)
from layer_lenses.training import set_seed


def quadratic_snapshot_epochs(total_epochs: int, n_points: int = 80) -> tuple[int, ...]:
    """Return snapshot epochs dense near initialization and sparser later."""
    if n_points < 2:
        return (0, total_epochs)
    vals = []
    for i in range(n_points):
        frac = i / (n_points - 1)
        vals.append(int(round((frac**2) * total_epochs)))
    vals.extend([0, total_epochs])
    return tuple(sorted(set(vals)))


def run_single_relu_seed(
    master_seed: int,
    *,
    dim: int,
    depth: int,
    n_train: int,
    hidden_dims: list[int],
    total_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    snapshot_epochs,
    bias: bool = False,
    threshold: float = 0.0,
    device: str = "cpu",
    show_progress: bool = True,
    lr_scheduler_eta_min_ratio: float = 0.05,
) -> dict[str, Any]:
    """Generate COB-ODT data, train one ReLU MLP seed, and return notebook-friendly state."""
    data_seed = master_seed
    init_seed = master_seed + 1000
    train_seed = master_seed + 2000

    x, y, tree, meta = generate_cob_odt_data(
        num_data=2 * n_train,
        dim=dim,
        depth=depth,
        seed=data_seed,
        threshold=threshold,
    )
    x_train, y_train = x[:n_train], y[:n_train]
    x_eval, y_eval = x[n_train:], y[n_train:]

    set_seed(init_seed)
    model = ReLUMLP(input_dim=dim, hidden_dims=hidden_dims, bias=bias)
    out = train_relu_mlp(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=ReLUTrainConfig(
            epochs=total_epochs,
            lr=lr,
            batch_size=batch_size,
            seed=train_seed,
            snapshot_epochs=tuple(snapshot_epochs),
            weight_decay=weight_decay,
            device=device,
            show_progress=show_progress,
            lr_scheduler_eta_min_ratio=lr_scheduler_eta_min_ratio,
        ),
    )

    return {
        "master_seed": master_seed,
        "seeds": {"data_seed": data_seed, "init_seed": init_seed, "train_seed": train_seed},
        "tree": tree,
        "meta": meta,
        "data": {"x_train": x_train, "y_train": y_train, "x_eval": x_eval, "y_eval": y_eval},
        "out": out,
    }


def collect_start_end_metrics(seed_result: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """Evaluate a ReLU run at its first and last checkpoint."""
    out = seed_result["out"]
    ckpts = out["checkpoint_snapshots"]
    epochs_sorted = sorted(ckpts.keys())
    e0, eN = epochs_sorted[0], epochs_sorted[-1]

    x_train = seed_result["data"]["x_train"]
    y_train = seed_result["data"]["y_train"]
    x_eval = seed_result["data"]["x_eval"]
    y_eval = seed_result["data"]["y_eval"]

    rows: list[dict[str, float | int | str]] = []
    for label, ep in [("start", e0), ("end", eN)]:
        model = checkpoint_model_from_state(out["model"], ckpts[ep])
        m_tr = evaluate_relu_mlp(model=model, x_eval=x_train, y_eval=y_train)
        m_te = evaluate_relu_mlp(model=model, x_eval=x_eval, y_eval=y_eval)
        rows.append(
            {
                "timepoint": label,
                "epoch": ep,
                "train_log_loss": float(m_tr["log_loss"]),
                "test_error": float(m_te["zero_one_loss"]),
            }
        )
    return rows


def first_layer_odt_alignment(
    *,
    state_dict: dict[str, torch.Tensor],
    tree: COBODTTree,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Compute first-layer ReLU weight alignment to COB-ODT decision vectors."""
    first_layer_wts = state_dict["hidden_layers.0.weight"].detach().cpu()
    odt_wts = torch.from_numpy(tree.w_list).float()
    row_norms = first_layer_wts.norm(dim=1)
    cos = (first_layer_wts @ odt_wts.T) / (
        row_norms.unsqueeze(1).clamp_min(1e-12)
        * odt_wts.norm(dim=1).unsqueeze(0).clamp_min(1e-12)
    )
    abs_cos = cos.abs()
    max_abs_cos, closest_odt = abs_cos.max(dim=1)
    closest_odt_np = closest_odt.numpy()
    node_level = np.floor(np.log2(closest_odt_np + 1)).astype(int)
    signed_closest_cos = cos[torch.arange(first_layer_wts.shape[0]), closest_odt]

    return {
        "first_layer_wts": first_layer_wts,
        "odt_wts": odt_wts,
        "row_norms": row_norms,
        "cos": cos,
        "abs_cos": abs_cos,
        "max_abs_cos": max_abs_cos,
        "closest_odt": closest_odt,
        "closest_odt_np": closest_odt_np,
        "node_level": node_level,
        "signed_closest_cos": signed_closest_cos,
    }


def plot_first_layer_odt_alignment(
    *,
    state_dict: dict[str, torch.Tensor],
    tree: COBODTTree,
    depth: int,
    epoch: int,
    seed: int | None = None,
    ax=None,
):
    """Plot first-layer norm vs max absolute cosine to ODT decision vectors."""
    alignment = first_layer_odt_alignment(state_dict=state_dict, tree=tree)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    level_names = {i: f"level {i}" for i in range(depth)}
    cmap = plt.get_cmap("viridis", depth)
    sc = ax.scatter(
        alignment["row_norms"].numpy(),
        alignment["max_abs_cos"].numpy(),
        c=alignment["node_level"],
        cmap=cmap,
        s=18,
        alpha=0.8,
        vmin=-0.5,
        vmax=depth - 0.5,
    )
    cbar = plt.colorbar(sc, ax=ax, ticks=range(depth))
    cbar.ax.set_yticklabels([level_names[i] for i in range(depth)])
    cbar.set_label("ODT level of argmax |cosine|")
    ax.set_xlabel("First-layer ReLU neuron weight norm")
    ax.set_ylabel("Max |cosine| similarity to ODT vectors")
    seed_text = f"Seed={seed}, " if seed is not None else ""
    ax.set_title(f"{seed_text}Epoch={epoch}: alignment by ODT level")
    ax.grid(alpha=0.2)
    return alignment, ax


def _layer_y_positions(width: int) -> np.ndarray:
    if width == 1:
        return np.array([0.5])
    return np.linspace(0.98, 0.02, width)


def plot_layered_relu_graph(
    *,
    state_dict: dict[str, torch.Tensor],
    tree: COBODTTree,
    depth: int,
    epoch: int,
    node_mask: np.ndarray | None = None,
    alignment_abs_threshold: float = 0.5,
    edge_abs_threshold: float = 0.2,
    figsize: tuple[float, float] = (14, 18),
    require_both_edge_endpoints_selected: bool = True,
):
    """Plot a layered hidden-node graph with ODT labels on first-layer neurons."""
    alignment = first_layer_odt_alignment(state_dict=state_dict, tree=tree)
    w0 = state_dict["hidden_layers.0.weight"].detach().cpu()
    w1 = state_dict["hidden_layers.1.weight"].detach().cpu()
    w2 = state_dict["hidden_layers.2.weight"].detach().cpu()
    wout = state_dict["output_layer.weight"].detach().cpu()

    if node_mask is None:
        node_mask = np.ones((w0.shape[0], 3), dtype=bool)
    else:
        node_mask = np.asarray(node_mask, dtype=bool)

    widths = [w0.shape[0], w1.shape[0], w2.shape[0], 1]
    x_positions = [0.0, 1.6, 3.2, 4.8]
    layer_names = ["hidden 1", "hidden 2", "hidden 3", "output"]

    node_pos = {}
    for layer_idx, width in enumerate(widths):
        ys = _layer_y_positions(width)
        for node_idx, y in enumerate(ys):
            node_pos[(layer_idx, node_idx)] = (x_positions[layer_idx], y)

    fig, ax = plt.subplots(figsize=figsize)
    edge_mats = [
        (w1, 0, 1),
        (w2, 1, 2),
        (wout, 2, 3),
    ]

    visible_edges = []
    for mat, src_layer, dst_layer in edge_mats:
        for dst_idx in range(mat.shape[0]):
            for src_idx in range(mat.shape[1]):
                weight = float(mat[dst_idx, src_idx])
                if abs(weight) < edge_abs_threshold:
                    continue

                src_selected = bool(node_mask[src_idx, src_layer])
                dst_selected = True if dst_layer == 3 else bool(node_mask[dst_idx, dst_layer])
                keep_edge = src_selected and dst_selected
                if not require_both_edge_endpoints_selected:
                    keep_edge = src_selected or dst_selected
                if keep_edge:
                    visible_edges.append((src_layer, src_idx, dst_layer, dst_idx, weight))

    max_visible_abs = max((abs(w) for *_, w in visible_edges), default=1.0)

    for src_layer, src_idx, dst_layer, dst_idx, weight in visible_edges:
        x0, y0 = node_pos[(src_layer, src_idx)]
        x1, y1 = node_pos[(dst_layer, dst_idx)]
        edge_color = "#d62728" if weight > 0 else "#1f77b4"
        edge_alpha = 0.15 + 0.65 * (abs(weight) / max_visible_abs)
        edge_lw = 0.4 + 2.5 * (abs(weight) / max_visible_abs)
        ax.plot([x0, x1], [y0, y1], color=edge_color, alpha=edge_alpha, linewidth=edge_lw, zorder=1)

    max_abs_cos = alignment["max_abs_cos"]
    signed_closest_cos = alignment["signed_closest_cos"]
    closest_odt = alignment["closest_odt"]

    for layer_idx, width in enumerate(widths):
        xs = [node_pos[(layer_idx, i)][0] for i in range(width)]
        ys = [node_pos[(layer_idx, i)][1] for i in range(width)]
        if layer_idx == 0:
            colors = []
            for i in range(width):
                if float(max_abs_cos[i]) < alignment_abs_threshold:
                    colors.append("#bdbdbd")
                elif float(signed_closest_cos[i]) >= 0:
                    colors.append("#d62728")
                else:
                    colors.append("#1f77b4")
        elif layer_idx in (1, 2):
            colors = ["#f2f2f2"] * width
        else:
            colors = ["#f7d774"]

        ax.scatter(
            xs,
            ys,
            s=260 if layer_idx < 3 else 520,
            c=colors,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

    for i in range(w0.shape[0]):
        x, y = node_pos[(0, i)]
        label = f"{int(closest_odt[i])}\n{float(signed_closest_cos[i]):+.2f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=5, zorder=4)

    for layer_idx in [1, 2]:
        for i in range(widths[layer_idx]):
            x, y = node_pos[(layer_idx, i)]
            ax.text(x, y, str(i), ha="center", va="center", fontsize=5, zorder=4)

    ax.text(*node_pos[(3, 0)], "out", ha="center", va="center", fontsize=8, weight="bold", zorder=4)
    for layer_idx, name in enumerate(layer_names):
        ax.text(
            x_positions[layer_idx],
            1.045,
            name,
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    ax.set_title(
        f"ReLU MLP connectivity at epoch {epoch}\n"
        f"First-layer color: signed cosine to closest ODT node; gray if |cos| < {alignment_abs_threshold}. "
        f"Edges shown if |weight| >= {edge_abs_threshold}.",
        fontsize=13,
    )
    ax.set_xlim(-0.45, 5.25)
    ax.set_ylim(-0.04, 1.08)
    ax.axis("off")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markeredgecolor="black",
            markersize=10,
            label="positive closest ODT cosine",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markersize=10,
            label="negative closest ODT cosine",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#bdbdbd",
            markeredgecolor="black",
            markersize=10,
            label=f"|cos| < {alignment_abs_threshold}",
        ),
        plt.Line2D([0], [0], color="#d62728", lw=2, label="positive ReLU edge weight"),
        plt.Line2D([0], [0], color="#1f77b4", lw=2, label="negative ReLU edge weight"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3)

    print(f"Visible edges: {len(visible_edges)} / {w1.numel() + w2.numel() + wout.numel()}")
    print(f"First-layer aligned nodes: {int((max_abs_cos >= alignment_abs_threshold).sum())} / {w0.shape[0]}")
    return alignment, visible_edges, fig, ax


def masked_relu_logits(
    model: ReLUMLP,
    x: np.ndarray,
    node_mask: np.ndarray,
    *,
    batch_size: int = 8192,
    device: str = "cpu",
) -> np.ndarray:
    """Run ReLU MLP while zeroing hidden activations where ``node_mask[row, layer] == 0``."""
    model = model.to(device)
    model.eval()

    keep = torch.as_tensor(node_mask.astype(bool), device=device)
    hidden_widths = [layer.out_features for layer in model.hidden_layers]
    num_hidden_layers = len(hidden_widths)

    if keep.ndim != 2:
        raise ValueError(f"node_mask must be 2D, got shape {node_mask.shape}.")
    if keep.shape[1] != num_hidden_layers:
        raise ValueError(
            "node_mask must have one column per hidden layer. "
            f"Expected {num_hidden_layers}, got {keep.shape[1]}."
        )
    if keep.shape[0] < max(hidden_widths):
        raise ValueError(
            f"node_mask has too few rows. Need at least {max(hidden_widths)}, "
            f"got {keep.shape[0]}."
        )

    x_t = torch.from_numpy(x).float().to(device)
    logits_batches = []

    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            h = x_t[start : start + batch_size]

            for layer_idx, layer in enumerate(model.hidden_layers):
                h = torch.relu(layer(h))
                layer_keep = keep[: hidden_widths[layer_idx], layer_idx].float()
                h = h * layer_keep.unsqueeze(0)

            logits = model.output_layer(h).squeeze(-1)
            logits_batches.append(logits.detach().cpu())

    return torch.cat(logits_batches).numpy()


def masked_accuracy_by_odt_leaf(
    *,
    model: ReLUMLP,
    node_mask: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
) -> pd.DataFrame:
    """Aggregate masked-model accuracy and log loss by reached ODT leaf."""
    logits = masked_relu_logits(
        model=model,
        x=x_eval,
        node_mask=node_mask,
        batch_size=batch_size,
        device=device,
    )
    y_eval_pm1 = y_eval.astype(np.int8)
    pred_pm1 = np.where(logits >= 0, 1, -1).astype(np.int8)
    per_example_log_loss = np.logaddexp(0.0, -y_eval_pm1.astype(np.float64) * logits)

    leaf_node = odt_leaf_ids_for_x(x_eval, tree)
    num_internal = tree.w_list.shape[0]

    rows = pd.DataFrame(
        {
            "leaf_node": leaf_node,
            "leaf_index": leaf_node - num_internal,
            "true_label": y_eval_pm1,
            "pred_label": pred_pm1,
            "correct": pred_pm1 == y_eval_pm1,
            "logit": logits,
            "log_loss": per_example_log_loss,
        }
    )

    return (
        rows.groupby(["leaf_node", "leaf_index", "true_label"], as_index=False)
        .agg(
            num_samples=("correct", "size"),
            num_correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_logit=("logit", "mean"),
            mean_abs_logit=("logit", lambda s: float(np.mean(np.abs(s)))),
        )
        .sort_values("leaf_node")
        .reset_index(drop=True)
    )


def neuron_leaf_responsibility_from_ablation(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    tree: COBODTTree,
    base_node_mask: np.ndarray | None = None,
    min_log_loss_delta: float = 0.05,
    min_relative_log_loss_delta: float | None = None,
    min_accuracy_drop: float | None = None,
    batch_size: int = 8192,
    device: str = "cpu",
) -> tuple[dict[tuple[int, int], dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Assign hidden neurons to ODT leaves by single-neuron ablation impact."""
    hidden_widths = [layer.out_features for layer in model.hidden_layers]
    num_hidden_layers = len(hidden_widths)
    max_width = max(hidden_widths)

    if base_node_mask is None:
        base_node_mask = np.ones((max_width, num_hidden_layers), dtype=int)
    else:
        base_node_mask = np.array(base_node_mask, dtype=int, copy=True)

    if base_node_mask.shape[1] != num_hidden_layers:
        raise ValueError(
            f"base_node_mask must have {num_hidden_layers} columns, got {base_node_mask.shape[1]}."
        )
    if base_node_mask.shape[0] < max_width:
        raise ValueError(
            f"base_node_mask must have at least {max_width} rows, got {base_node_mask.shape[0]}."
        )

    baseline_df = masked_accuracy_by_odt_leaf(
        model=model,
        node_mask=base_node_mask,
        x_eval=x_eval,
        y_eval=y_eval,
        tree=tree,
        batch_size=batch_size,
        device=device,
    ).rename(
        columns={
            "accuracy": "baseline_accuracy",
            "mean_log_loss": "baseline_mean_log_loss",
            "mean_logit": "baseline_mean_logit",
            "mean_abs_logit": "baseline_mean_abs_logit",
        }
    )

    responsibility: dict[tuple[int, int], dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []

    for layer_idx, width in enumerate(hidden_widths):
        for neuron_id in range(width):
            if base_node_mask[neuron_id, layer_idx] == 0:
                continue

            ablated_mask = base_node_mask.copy()
            ablated_mask[neuron_id, layer_idx] = 0

            ablated_df = masked_accuracy_by_odt_leaf(
                model=model,
                node_mask=ablated_mask,
                x_eval=x_eval,
                y_eval=y_eval,
                tree=tree,
                batch_size=batch_size,
                device=device,
            ).rename(
                columns={
                    "accuracy": "ablated_accuracy",
                    "mean_log_loss": "ablated_mean_log_loss",
                    "mean_logit": "ablated_mean_logit",
                    "mean_abs_logit": "ablated_mean_abs_logit",
                }
            )

            merged = baseline_df.merge(
                ablated_df[
                    [
                        "leaf_node",
                        "ablated_accuracy",
                        "ablated_mean_log_loss",
                        "ablated_mean_logit",
                        "ablated_mean_abs_logit",
                    ]
                ],
                on="leaf_node",
                how="inner",
            )
            merged["log_loss_delta"] = (
                merged["ablated_mean_log_loss"] - merged["baseline_mean_log_loss"]
            )
            merged["relative_log_loss_delta"] = merged["log_loss_delta"] / merged[
                "baseline_mean_log_loss"
            ].clip(lower=1e-12)
            merged["accuracy_drop"] = merged["baseline_accuracy"] - merged["ablated_accuracy"]

            selected = merged[merged["log_loss_delta"] >= min_log_loss_delta].copy()
            if min_relative_log_loss_delta is not None:
                selected = selected[
                    selected["relative_log_loss_delta"] >= min_relative_log_loss_delta
                ]
            if min_accuracy_drop is not None:
                selected = selected[selected["accuracy_drop"] >= min_accuracy_drop]

            selected = selected.sort_values("log_loss_delta", ascending=False).reset_index(drop=True)
            if selected.empty:
                responsibility[(layer_idx, neuron_id)] = {
                    "leaf_nodes": [],
                    "num_responsible_leaves": 0,
                    "total_weighted_log_loss_delta": 0.0,
                    "max_log_loss_delta": 0.0,
                    "details": [],
                }
                continue

            details = []
            for row in selected.to_dict(orient="records"):
                detail = {
                    "leaf_node": int(row["leaf_node"]),
                    "leaf_index": int(row["leaf_index"]),
                    "true_label": int(row["true_label"]),
                    "num_samples": int(row["num_samples"]),
                    "baseline_accuracy": float(row["baseline_accuracy"]),
                    "ablated_accuracy": float(row["ablated_accuracy"]),
                    "accuracy_drop": float(row["accuracy_drop"]),
                    "baseline_mean_log_loss": float(row["baseline_mean_log_loss"]),
                    "ablated_mean_log_loss": float(row["ablated_mean_log_loss"]),
                    "log_loss_delta": float(row["log_loss_delta"]),
                    "relative_log_loss_delta": float(row["relative_log_loss_delta"]),
                }
                details.append(detail)
                detail_rows.append({"layer": layer_idx, "neuron_id": neuron_id, **detail})

            responsibility[(layer_idx, neuron_id)] = {
                "leaf_nodes": [d["leaf_node"] for d in details],
                "num_responsible_leaves": len(details),
                "total_weighted_log_loss_delta": float(
                    np.average(selected["log_loss_delta"], weights=selected["num_samples"])
                ),
                "max_log_loss_delta": float(selected["log_loss_delta"].max()),
                "details": details,
            }

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            ["layer", "neuron_id", "log_loss_delta"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return responsibility, detail_df, baseline_df


def summarize_neuron_responsibility(
    responsibility: dict[tuple[int, int], dict[str, Any]],
) -> pd.DataFrame:
    """Return one compact row per neuron with at least one attributed ODT leaf."""
    rows = [
        {
            "layer": layer,
            "neuron_id": neuron_id,
            "leaf_nodes": payload["leaf_nodes"],
            "num_responsible_leaves": payload["num_responsible_leaves"],
            "max_log_loss_delta": payload["max_log_loss_delta"],
            "total_weighted_log_loss_delta": payload["total_weighted_log_loss_delta"],
        }
        for (layer, neuron_id), payload in responsibility.items()
        if payload["num_responsible_leaves"] > 0
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "layer",
                "neuron_id",
                "leaf_nodes",
                "num_responsible_leaves",
                "max_log_loss_delta",
                "total_weighted_log_loss_delta",
            ]
        )
    return pd.DataFrame(rows).sort_values(["layer", "neuron_id"]).reset_index(drop=True)


def _precompute_hidden_activation_masks_and_leaf_ids(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict[str, Any]:
    """Precompute hidden ReLU activation masks and ODT leaf ids for ``x_eval``."""
    precomputed = _precompute_hidden_activation_masks(
        model=model,
        x_eval=x_eval,
        batch_size=batch_size,
        device=device,
    )
    return {**precomputed, "leaf_ids": odt_leaf_ids_for_x(x_eval, tree)}


def _resolve_hidden_neuron_id(
    neuron_id: int | tuple[int, int],
    *,
    layer_widths: list[int],
    layer_offsets: list[int],
    total_hidden_neurons: int,
) -> tuple[int, int, int]:
    """Resolve a neuron id to (layer_idx, local_idx, global_idx)."""
    if isinstance(neuron_id, tuple):
        if len(neuron_id) != 2:
            raise ValueError("Tuple neuron_id must be (layer_idx, local_neuron_idx).")
        layer_idx, local_idx = int(neuron_id[0]), int(neuron_id[1])
        if layer_idx < 0 or layer_idx >= len(layer_widths):
            raise ValueError(
                f"layer_idx must be in [0, {len(layer_widths) - 1}], got {layer_idx}."
            )
        if local_idx < 0 or local_idx >= layer_widths[layer_idx]:
            raise ValueError(
                f"local neuron index must be in [0, {layer_widths[layer_idx] - 1}] "
                f"for layer {layer_idx}, got {local_idx}."
            )
        global_idx = layer_offsets[layer_idx] + local_idx
        return layer_idx, local_idx, global_idx

    global_idx = int(neuron_id)
    if global_idx < 0 or global_idx >= total_hidden_neurons:
        raise ValueError(
            f"Global neuron id must be in [0, {total_hidden_neurons - 1}], got {global_idx}."
        )
    layer_idx = max(
        i
        for i, offset in enumerate(layer_offsets)
        if global_idx >= offset
    )
    local_idx = global_idx - layer_offsets[layer_idx]
    return layer_idx, local_idx, global_idx


def _precompute_hidden_activation_masks(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict[str, Any]:
    """Precompute boolean hidden-layer activation masks for ``x_eval``."""
    model = model.to(device)
    model.eval()

    x_t = torch.from_numpy(x_eval).float().to(device)
    hidden_masks_per_layer: list[list[torch.Tensor]] = [
        [] for _ in range(len(model.hidden_layers))
    ]

    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            h = x_t[start : start + batch_size]
            for layer_idx, layer in enumerate(model.hidden_layers):
                pre_act = layer(h)
                hidden_masks_per_layer[layer_idx].append((pre_act > 0).detach().cpu())
                h = torch.relu(pre_act)

    activation_masks = [torch.cat(parts, dim=0).numpy() for parts in hidden_masks_per_layer]
    layer_widths = [layer.out_features for layer in model.hidden_layers]
    layer_offsets = np.cumsum([0, *layer_widths[:-1]]).tolist()
    total_hidden_neurons = int(sum(layer_widths))

    return {
        "activation_masks": activation_masks,
        "layer_widths": layer_widths,
        "layer_offsets": layer_offsets,
        "total_hidden_neurons": total_hidden_neurons,
    }


def active_point_mask_for_neuron(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    neuron_id: int | tuple[int, int],
    batch_size: int = 8192,
    device: str = "cpu",
    precomputed: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return boolean mask of points in ``x_eval`` that activate a hidden neuron.

    Args:
        model: Checkpoint-loaded ReLU model.
        x_eval: Evaluation inputs of shape ``(n, d)``.
        neuron_id: Either global hidden neuron id (0..total_hidden-1) or tuple
            ``(layer_idx, local_neuron_idx)``.
        precomputed: Optional cache from ``_precompute_hidden_activation_masks``
            for repeated queries.

    Returns:
        Boolean array of shape ``(n,)``; ``True`` means the input activates that neuron.
    """
    if precomputed is None:
        precomputed = _precompute_hidden_activation_masks(
            model=model,
            x_eval=x_eval,
            batch_size=batch_size,
            device=device,
        )

    layer_idx, local_idx, _ = _resolve_hidden_neuron_id(
        neuron_id,
        layer_widths=precomputed["layer_widths"],
        layer_offsets=precomputed["layer_offsets"],
        total_hidden_neurons=precomputed["total_hidden_neurons"],
    )
    return precomputed["activation_masks"][layer_idx][:, local_idx].astype(bool)


def count_active_points_for_neuron_leaf(
    *,
    model: ReLUMLP,
    neuron_id: int | tuple[int, int],
    leaf_node_id: int,
    x_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
    precomputed: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    """Count how many points in a given ODT leaf activate a given hidden neuron.

    Args:
        model: Checkpoint-loaded ReLU model.
        neuron_id: Either global hidden neuron id (0..total_hidden-1) or
            tuple ``(layer_idx, local_neuron_idx)``.
        leaf_node_id: ODT label leaf id in heap indexing (e.g. 31..62 for depth 5).
        x_eval: Input data used for counting.
        tree: COB-ODT tree used to assign leaves.
        precomputed: Optional cache from
            ``_precompute_hidden_activation_masks_and_leaf_ids`` to reuse in loops.

    Returns:
        Dictionary with layer/local/global neuron id, total points in leaf, active points,
        and active fraction.
    """
    if precomputed is None:
        precomputed = _precompute_hidden_activation_masks_and_leaf_ids(
            model=model,
            x_eval=x_eval,
            tree=tree,
            batch_size=batch_size,
            device=device,
        )

    layer_idx, local_idx, global_idx = _resolve_hidden_neuron_id(
        neuron_id,
        layer_widths=precomputed["layer_widths"],
        layer_offsets=precomputed["layer_offsets"],
        total_hidden_neurons=precomputed["total_hidden_neurons"],
    )

    leaf_ids = precomputed["leaf_ids"]
    leaf_mask = leaf_ids == int(leaf_node_id)
    total_leaf_points = int(leaf_mask.sum())
    if total_leaf_points == 0:
        active_points = 0
    else:
        active_points = int(precomputed["activation_masks"][layer_idx][leaf_mask, local_idx].sum())
    active_fraction = float(active_points / total_leaf_points) if total_leaf_points > 0 else 0.0

    return {
        "leaf_node_id": int(leaf_node_id),
        "layer": int(layer_idx),
        "local_neuron_id": int(local_idx),
        "global_neuron_id": int(global_idx),
        "total_points_in_leaf": total_leaf_points,
        "active_points_in_leaf": active_points,
        "active_fraction_in_leaf": active_fraction,
    }


def summarize_neuron_leaf_activity(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
) -> pd.DataFrame:
    """Return activity summary for all hidden neurons across all reached ODT leaves."""
    precomputed = _precompute_hidden_activation_masks_and_leaf_ids(
        model=model,
        x_eval=x_eval,
        tree=tree,
        batch_size=batch_size,
        device=device,
    )
    unique_leaf_ids = np.unique(precomputed["leaf_ids"]).tolist()
    rows: list[dict[str, int | float]] = []
    for global_neuron_id in range(precomputed["total_hidden_neurons"]):
        for leaf_node_id in unique_leaf_ids:
            rows.append(
                count_active_points_for_neuron_leaf(
                    model=model,
                    neuron_id=global_neuron_id,
                    leaf_node_id=int(leaf_node_id),
                    x_eval=x_eval,
                    tree=tree,
                    batch_size=batch_size,
                    device=device,
                    precomputed=precomputed,
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["layer", "local_neuron_id", "leaf_node_id"],
        ignore_index=True,
    )


def neuron_leaf_activity_dict(
    neuron_leaf_activity_summary: pd.DataFrame,
    *,
    key_mode: str = "global",
) -> tuple[dict[int | tuple[int, int], np.ndarray], np.ndarray]:
    """Convert neuron-leaf activity table into neuron -> leaf-activity vector mapping.

    Args:
        neuron_leaf_activity_summary: DataFrame produced by
            ``summarize_neuron_leaf_activity`` (or equivalent), expected to contain:
            - ``leaf_node_id``
            - ``active_points_in_leaf``
            - and either:
              - ``global_neuron_id`` (for ``key_mode='global'``), or
              - ``layer`` and ``local_neuron_id`` (for ``key_mode='layer_local'``).
        key_mode: Key format for the output dictionary:
            - ``'global'`` -> keys are global neuron ids (int).
            - ``'layer_local'`` -> keys are ``(layer, local_neuron_id)`` tuples.

    Returns:
        (mapping, leaf_order)
            mapping: dict where each key is a neuron id and each value is a numpy array
                of length ``len(leaf_order)``, with entries equal to
                ``active_points_in_leaf`` in that leaf.
            leaf_order: sorted numpy array of leaf node ids used as vector order.
    """
    df = neuron_leaf_activity_summary.copy()
    if df.empty:
        return {}, np.array([], dtype=np.int64)

    required_common = {"leaf_node_id", "active_points_in_leaf"}
    missing_common = required_common.difference(df.columns)
    if missing_common:
        raise ValueError(
            "neuron_leaf_activity_summary missing required columns: "
            f"{sorted(missing_common)}"
        )

    if key_mode not in {"global", "layer_local"}:
        raise ValueError(f"key_mode must be 'global' or 'layer_local', got {key_mode!r}.")

    if key_mode == "global":
        if "global_neuron_id" not in df.columns:
            raise ValueError(
                "For key_mode='global', neuron_leaf_activity_summary must include "
                "'global_neuron_id'."
            )
    else:
        required_pair = {"layer", "local_neuron_id"}
        missing_pair = required_pair.difference(df.columns)
        if missing_pair:
            raise ValueError(
                "For key_mode='layer_local', neuron_leaf_activity_summary missing columns: "
                f"{sorted(missing_pair)}"
            )

    leaf_order = np.sort(df["leaf_node_id"].astype(np.int64).unique())
    leaf_to_pos = {leaf_id: i for i, leaf_id in enumerate(leaf_order)}

    mapping: dict[int | tuple[int, int], np.ndarray] = {}

    if key_mode == "global":
        neuron_keys = sorted(df["global_neuron_id"].astype(int).unique().tolist())
        for neuron_id in neuron_keys:
            vec = np.zeros(len(leaf_order), dtype=np.int64)
            sub = df[df["global_neuron_id"].astype(int) == neuron_id]
            for row in sub.itertuples(index=False):
                pos = leaf_to_pos[int(row.leaf_node_id)]
                vec[pos] = int(row.active_points_in_leaf)
            mapping[int(neuron_id)] = vec
    else:
        subkeys = (
            df[["layer", "local_neuron_id"]]
            .astype(int)
            .drop_duplicates()
            .sort_values(["layer", "local_neuron_id"])
        )
        for rowkey in subkeys.itertuples(index=False):
            key = (int(rowkey.layer), int(rowkey.local_neuron_id))
            vec = np.zeros(len(leaf_order), dtype=np.int64)
            sub = df[
                (df["layer"].astype(int) == key[0])
                & (df["local_neuron_id"].astype(int) == key[1])
            ]
            for row in sub.itertuples(index=False):
                pos = leaf_to_pos[int(row.leaf_node_id)]
                vec[pos] = int(row.active_points_in_leaf)
            mapping[key] = vec

    return mapping, leaf_order
