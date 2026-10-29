"""A slim model for the Korean head: only the blocks and SAE features it reads.

The Korean head uses SAEs up to `max_layer` and a few dozen of their 16k
features each, so build() keeps Gemma's blocks below the deepest hook (bf16)
and the encoder columns of the head's features, and saves them to
work/ko_model (~4 GB instead of the 11 GB fp32 detector). KoModel.extract()
returns the same activations as AuditableDetector.extract() for those
features, so score.ko_detect() works on either.

The deepest hook reads the input of block `max_layer`, which is the output of
the last kept block, so it is captured at the input of the final norm and the
block itself is dropped.
"""

import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work", "ko_model")
MAX_TOKENS = 1024


def slice_sae(sae, local: torch.Tensor) -> dict[str, torch.Tensor]:
    """The encoder parameters of a JumpReLUSAE for the features `local` only."""
    return {
        "W_enc": sae.W_enc.data[:, local].float().cpu().contiguous(),
        "b_enc": sae.b_enc.data[local].float().cpu(),
        "threshold": torch.exp(sae.log_threshold.data[local]).float().cpu(),
        "b_dec": (sae.b_dec.data if sae.use_pre_enc_bias else torch.zeros_like(sae.b_dec.data)).float().cpu(),
    }


def encode(h: torch.Tensor, s: dict[str, torch.Tensor]) -> torch.Tensor:
    """JumpReLUSAE.encode restricted to the sliced features."""
    pre = F.relu((h.float() - s["b_dec"]) @ s["W_enc"] + s["b_enc"])
    return pre * (pre > s["threshold"])


def build(detector, head: dict, out: str = MODEL_DIR) -> None:
    """Saves the slim model for `head` from a loaded full detector (score.load)."""
    from score import D_SAE, HOOK_LAYERS

    n_blocks = head["max_layer"]
    lm = detector.extractor.model
    lm.model.layers = lm.model.layers[:n_blocks]
    lm.config.num_hidden_layers = n_blocks
    lm.config.layer_types = lm.config.layer_types[:n_blocks]
    lm.save_pretrained(os.path.join(out, "lm"))
    detector.extractor.tokenizer.save_pretrained(os.path.join(out, "lm"))

    ids = torch.tensor([f["id"] for f in head["features"]])
    tensors = {}
    for i, layer in enumerate(HOOK_LAYERS):
        if layer > n_blocks:
            continue
        local = ids[ids // D_SAE == i] % D_SAE
        for k, v in slice_sae(detector.saes[i], local).items():
            tensors[f"{layer}.{k}"] = v
        tensors[f"{layer}.ids"] = local + i * D_SAE
    save_file(tensors, os.path.join(out, "saes.safetensors"))
    json.dump({"feature_ids": ids.tolist(), "max_layer": n_blocks}, open(os.path.join(out, "meta.json"), "w"))


def matches(head: dict, path: str = MODEL_DIR) -> bool:
    """Whether a slim model was built for exactly this head's features."""
    meta = os.path.join(path, "meta.json")
    return os.path.exists(meta) and json.load(open(meta))["feature_ids"] == [f["id"] for f in head["features"]]


class KoModel:
    """Loads work/ko_model; extract() mirrors AuditableDetector.extract()."""

    def __init__(self, path: str = MODEL_DIR):
        from score import D_SAE, HOOK_LAYERS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(path, "lm"))
        self.lm = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(path, "lm"), dtype=torch.bfloat16)
        self.lm.lm_head = torch.nn.Identity()
        self.lm.to(self.device).eval()
        t = load_file(os.path.join(path, "saes.safetensors"), device=self.device)
        self.layers = sorted({int(k.split(".")[0]) for k in t})
        self.saes = {layer: {k: t[f"{layer}.{k}"] for k in ("ids", "W_enc", "b_enc", "threshold", "b_dec")} for layer in self.layers}
        self.width = len(HOOK_LAYERS) * D_SAE

    def _modules(self) -> dict[int, torch.nn.Module]:
        """Module whose input is the residual stream entering each hooked block."""
        blocks = self.lm.model.layers
        return {layer: blocks[layer] if layer < len(blocks) else self.lm.model.norm for layer in self.layers}

    @torch.no_grad()
    def extract(self, text: str):
        captured = {}

        def hook(mod, args, kwargs, layer):
            captured[layer] = args[0] if args else kwargs["hidden_states"]

        handles = [m.register_forward_pre_hook(lambda mod, a, kw, layer=layer: hook(mod, a, kw, layer), with_kwargs=True)
                   for layer, m in self._modules().items()]
        try:
            enc = self.tokenizer(text, max_length=MAX_TOKENS, truncation=True, return_tensors="pt",
                                 return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            self.lm(**{k: v.to(self.device) for k, v in enc.items()})
        finally:
            for h in handles:
                h.remove()

        cols, acts = [], []
        for layer in self.layers:
            acts.append(encode(captured[layer][0], self.saes[layer]))
            cols.append(self.saes[layer]["ids"])
        acts, cols = torch.cat(acts, 1).cpu(), torch.cat(cols).cpu()  # [seq, n_features], global ids
        x = torch.zeros(1, self.width)
        x[0, cols] = acts.max(0).values
        t, k = acts.nonzero(as_tuple=True)
        token_feats = torch.sparse_coo_tensor(torch.stack([t, cols[k]]), acts[t, k], (acts.shape[0], self.width))
        return SimpleNamespace(x=x, token_feats=token_feats, tokens=[text[a:b] for a, b in offsets])
