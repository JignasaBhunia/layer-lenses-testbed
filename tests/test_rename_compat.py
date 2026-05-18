import pickle

from layer_lenses.relu_mlp import ReLUMLP


def test_old_first_experiment_pickle_module_path_resolves() -> None:
    loaded_class = pickle.loads(b"cfirst_experiment.relu_mlp\nReLUMLP\n.")

    assert loaded_class is ReLUMLP
