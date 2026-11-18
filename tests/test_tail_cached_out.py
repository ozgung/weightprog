import numpy as np

from weightprog.construct import LogitPath, TailState


def test_captured_mlp_out_matches_recompute():
    emb = np.array([[1.0, 0.3], [0.9, 0.2]])
    path = LogitPath(0, np.ones(2), np.ones(2), 0.5, lambda ids: emb[ids], softcap=2)
    acts = np.array([[1.0, 0.5], [0.7, 2.0]])
    weights = np.array([[0.1, 0.4], [0.5, 0.8]])
    middle = np.array([[0.5, 0.9], [-0.5, 1.0]])
    recomputed = TailState(0, True, middle, acts, None, weights, path)
    captured = TailState(0, True, middle, acts, None, weights, path,
                         captured_mlp_out=acts @ weights.T)
    delta = np.array([[0.42, 0.06], [0.3, 0.7]])
    np.testing.assert_allclose(recomputed.logits([0, 1], delta), captured.logits([0, 1], delta))


if __name__ == "__main__":
    test_captured_mlp_out_matches_recompute()
    print("cached MLP output check passed")
