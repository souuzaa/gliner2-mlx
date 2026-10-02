#!/usr/bin/env python3
"""Convert a GLiNER2 span checkpoint (e.g. fastino/GLiNER2.5-Decide) to MLX.

Produces a self-contained Hugging Face repo in the style of mlx-community/clef-flash-4bit:

    model.safetensors          MLX weights (encoder quantized, heads in --dtype)
    config.json                gliner2 config + inlined encoder config + quantization
    tokenizer.json, ...        original tokenizer files (unchanged)
    gliner2_mlx.py             torch-free loader / inference code
    README.md                  model card

Examples:
    python convert.py                                  # 4-bit, group size 64
    python convert.py --q-bits 8
    python convert.py --no-quantize --dtype bfloat16
    python convert.py --q-bits 4 --upload-repo mlx-community/GLiNER2.5-Decide-4bit
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gliner2_mlx  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def default_mlx_path(hf_path: str, quantize: bool, bits: int, dtype: str) -> Path:
    name = hf_path.rstrip("/").split("/")[-1]
    suffix = f"{bits}bit" if quantize else {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[dtype]
    return Path("mlx_models") / f"{name}-{suffix}"


def convert(hf_path: str, mlx_path: Path, quantize: bool, bits: int, group_size: int, dtype: str) -> dict:
    src = Path(hf_path)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(hf_path, allow_patterns=["*.json", "*.safetensors", "*.md"]))

    config, encoder_config = gliner2_mlx._read_configs(src)
    print(f"[INFO] Loading {hf_path} ({config.get('architecture', 'span')}, {encoder_config['model_type']})")
    model = gliner2_mlx.build_model(config, encoder_config, gliner2_mlx.load_weights(src), DTYPES[dtype])

    if quantize:
        print(f"[INFO] Quantizing encoder to {bits}-bit (group size {group_size}); heads stay {dtype}")
        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=gliner2_mlx.quantization_predicate)

    mlx_path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(mlx_path / "model.safetensors"), weights, metadata={"format": "mlx"})

    out_config = {k: v for k, v in config.items() if not k.startswith("_")}
    out_config["encoder_config"] = {k: v for k, v in encoder_config.items() if not k.startswith("_")}
    out_config["torch_dtype"] = dtype
    if quantize:
        out_config["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
    (mlx_path / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")

    for name in TOKENIZER_FILES:
        shutil.copy(src / name, mlx_path / name)
    shutil.copy(HERE / "gliner2_mlx.py", mlx_path / "gliner2_mlx.py")

    n_params = sum(v.size for k, v in weights.items() if not k.endswith((".scales", ".biases")))
    size_mb = (mlx_path / "model.safetensors").stat().st_size / 1e6
    print(f"[INFO] Saved {mlx_path} ({size_mb:.0f} MB on disk)")
    return {"size_mb": size_mb, "params": n_params}


SMOKE_TEXT = "My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?"
SMOKE_TASKS = {"intent": ["order_status", "refund_request", "cancel_subscription", "login_problem", "other"]}


def smoke_test(mlx_path: Path) -> None:
    model = gliner2_mlx.load(mlx_path)
    print("[INFO] Smoke test:", model.classify_text(SMOKE_TEXT, SMOKE_TASKS, include_confidence=True))


MODEL_CARD = """---
license: apache-2.0
library_name: mlx
base_model: {base}
base_model_relation: {relation}
pipeline_tag: text-classification
language:
- en
tags:
- mlx
- gliner2
- deberta-v2
- text-classification
- zero-shot-classification
- named-entity-recognition
- custom-code
---

# {repo}

[{base}](https://huggingface.co/{base}) converted to MLX ({variant}) for Apple Silicon.

GLiNER2.5-Decide is a 340M-parameter schema-driven classifier (DeBERTa-v3-large encoder + GLiNER2 heads):
pass any label set at call time and get a decision in a single forward pass, no generated tokens.
**It is not a language model**: `mlx_lm` / `mlx_vlm` cannot load it. Use the bundled `gliner2_mlx.py`,
which runs the encoder and the GLiNER2 heads and reproduces the `gliner2` pre/post-processing.

## Usage

```bash
pip install mlx tokenizers huggingface_hub   # no torch needed
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("{repo}")
sys.path.insert(0, path)
import gliner2_mlx

model = gliner2_mlx.load(path)
model.classify_text(
    "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
    {{
        "sentiment": ["positive", "negative", "mixed", "neutral"],
        "aspects": {{
            "labels": ["battery", "keyboard", "screen", "camera", "price", "support"],
            "multi_label": True,
            "cls_threshold": 0.4,
        }},
    }},
)
# {{'sentiment': ..., 'aspects': [...]}}
```

The task syntax is the same as `gliner2`'s `classify_text`: a list of labels, a `{{label: description}}` dict,
or a config dict with `labels`, `multi_label`, `cls_threshold`, `prompt`, `examples`, `class_act`.
Pass `include_confidence=True` for scores, or use `model.predict(text, tasks)` for the full
probability distribution of every task. `batch_classify_text(texts, tasks, batch_size=8)` batches inputs.

Entity extraction (the GLiNER2 span head) is also available:

```python
model.extract_entities("Tim Cook announced the new iPhone in Cupertino.", ["person", "product", "location"])
```

See the [original model card](https://huggingface.co/{base}) for the full set of examples.

## Limitations

- **Not a chat model.** Only `gliner2_mlx.py` can run it.
- **Classification and entities only.** `gliner2` relation extraction (`extract_relations`) and
  structured JSON extraction (`extract_json`) are not ported; `*_long` chunking helpers are not ported either.
- **Custom code.** Like the clef MLX ports, this repo ships Python (`gliner2_mlx.py`) that you import and run.
  Read it before use if that matters in your environment.

## Conversion

- Encoder: {encoder_note}
- Heads (classifier, count, span): {dtype}, unquantized.
- Relative-position embeddings: {dtype}, unquantized.
- Tokenizer files copied unchanged. Converted with [`convert.py`](https://github.com/souuzaa/gliner2-mlx) (mlx {mlx_version}); parity and evaluation scripts are in the same repo.
{quality}
## License

Apache-2.0, following [{base}](https://huggingface.co/{base}).
"""


def write_model_card(mlx_path: Path, base: str, repo: str, quantize: bool, bits: int, group_size: int,
                     dtype: str, quality: str = "") -> None:
    variant = f"{bits}-bit" if quantize else dtype
    encoder_note = (
        f"Linear layers and word embeddings quantized to {bits}-bit (affine, group size {group_size})."
        if quantize else f"{dtype}, unquantized."
    )
    (mlx_path / "README.md").write_text(MODEL_CARD.format(
        base=base, repo=repo, variant=variant, relation="quantized",
        encoder_note=encoder_note, dtype=dtype, mlx_version=mx.__version__, quality=quality,
    ))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", default="fastino/GLiNER2.5-Decide", help="HF repo id or local dir of the gliner2 checkpoint")
    p.add_argument("--mlx-path", type=Path, help="output dir (default: mlx_models/<name>-<variant>)")
    p.add_argument("-q", "--quantize", action=argparse.BooleanOptionalAction, default=True, help="quantize the encoder (default: on)")
    p.add_argument("--q-bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8))
    p.add_argument("--q-group-size", type=int, default=64, choices=(32, 64, 128))
    p.add_argument("--dtype", default="bfloat16", choices=tuple(DTYPES), help="dtype of unquantized weights")
    p.add_argument("--quality", type=Path, help="markdown file appended to the model card (e.g. from evaluate.py)")
    p.add_argument("--upload-repo", help="push the result to this HF repo id (e.g. mlx-community/GLiNER2.5-Decide-4bit)")
    args = p.parse_args()

    mlx_path = args.mlx_path or default_mlx_path(args.hf_path, args.quantize, args.q_bits, args.dtype)
    convert(args.hf_path, mlx_path, args.quantize, args.q_bits, args.q_group_size, args.dtype)
    repo = args.upload_repo or f"mlx-community/{mlx_path.name}"
    quality = "\n" + args.quality.read_text().strip() + "\n" if args.quality else ""
    write_model_card(mlx_path, args.hf_path, repo, args.quantize, args.q_bits, args.q_group_size, args.dtype, quality)
    smoke_test(mlx_path)

    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload_repo, exist_ok=True)
        api.upload_folder(folder_path=str(mlx_path), repo_id=args.upload_repo, repo_type="model")
        print(f"[INFO] Uploaded to https://huggingface.co/{args.upload_repo}")


if __name__ == "__main__":
    main()
