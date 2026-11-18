"""Checks runnable without an external test runner."""
from types import SimpleNamespace

import numpy as np

from weightprog.construct import cv_gate, ridge
from weightprog.evaluate import run
from weightprog.probe import decision_from_text
from weightprog.target import Case


def test_multioutput_ridge():
    rng = np.random.default_rng(11)
    for n, d in [(5, 11), (11, 5)]:
        x = rng.normal(size=(n, d))
        y = rng.normal(size=(n, 3))
        w = rng.uniform(0.1, 2, size=n)
        actual = ridge(x, y, 0.7, w)
        expected = np.stack([ridge(x, target, 0.7, w) for target in y.T], axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_pair_groups_do_not_leak():
    x = np.repeat(np.eye(3), 2, axis=0)
    y = np.repeat([1, 0, 1], 2)
    result = cv_gate(x, y, lams=(1,), folds=3, groups=["a", "a", "b", "b", "c", "c"])
    np.testing.assert_array_equal(result[1]["oof_gate"], np.zeros(6))


def test_preserved_generation_ids():
    from contextlib import nullcontext

    class Tok:
        def encode(self, text):
            # Re-tokenization could merge "." with a leading "__".
            return {"M.": [1, 2], "M.__": [1, 5, 6]}[text]

        def decode(self, ids):
            return "".join({1: "M", 2: ".", 3: "__", 4: "fields"}[i] for i in ids)

        def token_to_id(self, text):
            return None

    tok = Tok()
    backend = SimpleNamespace(tokenizer=tok, edits=lambda edit: nullcontext(),
                              generate=lambda *a, **kw: [SimpleNamespace(tokens=[3, 4])])
    case = Case("trace", prompt="M.", good="^__fields")
    row = run(backend, [case], draws=1, include_token_ids=True)[0]
    decision = decision_from_text(tok, case.prompt, row["text"], r"fields",
                                  generated_ids=row["token_ids"])
    assert decision.prefix_ids == [1, 2, 3]
    assert decision.bad_id == 4
    assert row["label"] == "good"
    assert decision_from_text(tok, case.prompt, row["text"], r"(missing)?fields", group=1,
                              generated_ids=row["token_ids"]) is None
    assert "token_ids" not in run(backend, [case], draws=1)[0]


if __name__ == "__main__":
    test_multioutput_ridge()
    test_pair_groups_do_not_leak()
    test_preserved_generation_ids()
    print("3 checks passed")
