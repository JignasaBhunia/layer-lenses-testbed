import numpy as np

from first_experiment.odt import (
    build_default_cob_odt_tree,
    generate_cob_odt_data,
    samples_reaching_node,
)


def test_cob_odt_shapes_and_label_domain() -> None:
    x, y, tree, meta = generate_cob_odt_data(
        num_data=1024,
        dim=16,
        depth=3,
        seed=365,
        threshold=0.0,
    )

    assert x.shape == (meta["num_kept"], 16)
    assert y.shape == (meta["num_kept"],)
    assert set(np.unique(y)).issubset({-1, 1})
    assert tree.w_list.shape == (7, 16)
    assert tree.b_list.shape == (7,)
    assert tree.leaf_labels.shape == (8,)


def test_hyperplanes_are_orthonormal_for_default_tree() -> None:
    rng = np.random.default_rng(123)
    tree = build_default_cob_odt_tree(dim=20, depth=4, rng=rng)
    gram = tree.w_list @ tree.w_list.T
    assert np.allclose(gram, np.eye(15), atol=1e-6)


def test_sibling_leaf_labels_are_opposite() -> None:
    rng = np.random.default_rng(7)
    tree = build_default_cob_odt_tree(dim=20, depth=3, rng=rng)

    for i in range(0, tree.leaf_labels.shape[0], 2):
        assert tree.leaf_labels[i] == -tree.leaf_labels[i + 1]


def test_seed_reproducibility() -> None:
    out1 = generate_cob_odt_data(num_data=512, dim=12, depth=3, seed=111)
    out2 = generate_cob_odt_data(num_data=512, dim=12, depth=3, seed=111)

    x1, y1, tree1, meta1 = out1
    x2, y2, tree2, meta2 = out2

    assert np.allclose(x1, x2)
    assert np.array_equal(y1, y2)
    assert np.allclose(tree1.w_list, tree2.w_list)
    assert np.allclose(tree1.b_list, tree2.b_list)
    assert np.array_equal(tree1.leaf_labels, tree2.leaf_labels)
    assert meta1 == meta2


def test_samples_reaching_root_returns_all_points() -> None:
    x, _, tree, meta = generate_cob_odt_data(
        num_data=600, dim=12, depth=3, seed=8, threshold=0.0
    )
    subset = samples_reaching_node(x=x, tree=tree, node_id=0)
    assert subset.shape[0] == x.shape[0]


def test_samples_reaching_leaf_partition_covers_all_points() -> None:
    x, _, tree, meta = generate_cob_odt_data(
        num_data=700, dim=12, depth=3, seed=19, threshold=0.0
    )
    depth = int(meta["depth"])
    num_internal = int(meta["num_internal_nodes"])
    num_leaf = int(meta["num_leaf_nodes"])

    union_mask = np.zeros(x.shape[0], dtype=bool)
    for leaf_offset in range(num_leaf):
        _, mask = samples_reaching_node(
            x=x,
            tree=tree,
            node_id=num_internal + leaf_offset,
            return_mask=True,
        )
        assert np.all(~(union_mask & mask))
        union_mask |= mask
    assert np.all(union_mask)


def test_samples_reaching_node_invalid_node_id_raises() -> None:
    x, _, tree, meta = generate_cob_odt_data(
        num_data=100, dim=8, depth=3, seed=21, threshold=0.0
    )
    depth = int(meta["depth"])
    num_total_nodes = 2 ** (depth + 1) - 1
    try:
        samples_reaching_node(x=x, tree=tree, node_id=num_total_nodes)
        assert False, "Expected ValueError for out-of-range node_id."
    except ValueError:
        pass
