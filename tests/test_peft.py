"""export_peft: a PEFT-loaded adapter adds exactly B @ A, and the adapted model equals the backend under edits."""

import json

import numpy as np
import pytest

from weightprog.edits import EditSet, ModuleEdit, export_peft
from weightprog.names import module_name


def test_peft_adapter_equals_delta(tiny_dir, tmp_path, torch_be, sample_ids):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import Gemma4ForConditionalGeneration

    rng = np.random.default_rng(2)
    es = EditSet([ModuleEdit(module_name(5, "down_proj"), rng.standard_normal((64, 2)), rng.standard_normal((2, 96))),
                  ModuleEdit(module_name(0, "q_proj"), rng.standard_normal((64, 3)) * 0.2, rng.standard_normal((3, 64))),
                  ModuleEdit(module_name(1, "k_proj"), rng.standard_normal((32, 1)) * 0.2, rng.standard_normal((1, 64)))])
    ad = export_peft(es, tmp_path / "adapter", base_model=str(tiny_dir))
    cfg = json.loads((ad / "adapter_config.json").read_text())
    assert cfg["r"] == 3 and cfg["lora_alpha"] == 3 and len(cfg["rank_pattern"]) == 2
    assert cfg["task_type"] == "CAUSAL_LM"  # the competition's adapter format and the study's adapter verifier

    base = Gemma4ForConditionalGeneration.from_pretrained(tiny_dir, dtype=torch.float32, attn_implementation="eager").eval()
    W0 = {e.module: dict(base.named_parameters())[e.module].detach().clone().numpy() for e in es}
    model = peft.PeftModel.from_pretrained(base, str(ad)).eval()
    merged = model.merge_and_unload()
    params = dict(merged.named_parameters())
    for e in es:
        np.testing.assert_allclose(params[e.module].detach().numpy(), W0[e.module] + e.delta(), rtol=0, atol=2e-6)
    with torch.no_grad():
        lg = merged(input_ids=torch.tensor([sample_ids]), use_cache=False).logits[0].float()
        lp_peft = (lg - torch.logsumexp(lg, -1, keepdim=True)).numpy()
    pos = list(range(len(sample_ids)))
    with torch_be.edits(es):
        lp_be = torch_be.logprobs(sample_ids, pos)
    np.testing.assert_allclose(lp_peft, lp_be, atol=1e-3)  # eager full forward vs sdpa cached forward
