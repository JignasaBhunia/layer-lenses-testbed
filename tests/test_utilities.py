import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from layer_lenses.utilities import (
    complete_binary_internal_node_positions,
    plot_complete_binary_tree_values,
)


def test_complete_binary_internal_node_positions_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="complete binary"):
        complete_binary_internal_node_positions(30)


def test_plot_complete_binary_tree_values_requires_matching_length() -> None:
    with pytest.raises(ValueError, match="length"):
        plot_complete_binary_tree_values(
            np.arange(7, dtype=float), labels=("a",) * 6
        )


def test_plot_complete_binary_tree_values_smoke_depth3_and_depth5() -> None:
    n = 31
    vals = np.linspace(0, 10, n)
    fig, ax = plot_complete_binary_tree_values(vals)
    assert fig is not None
    assert ax.get_xlim()[0] < 0

    fig2, ax2 = plot_complete_binary_tree_values(
        np.arange(7, dtype=float), use_dlgn_panel_geometry=True
    )
    assert fig2.get_figwidth() == pytest.approx(12.0)


def test_plot_complete_binary_tree_values_all_nan() -> None:
    a = np.full(7, np.nan)
    fig, _ = plot_complete_binary_tree_values(a)
    assert fig is not None
