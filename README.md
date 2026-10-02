# GLiNER2 → MLX converter

Converts [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (and other GLiNER2 *span*
checkpoints with a DeBERTa-v3 encoder) into an MLX repo laid out like
[mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit): quantized weights, the original
tokenizer, a torch-free loader (`gliner2_mlx.py`) and a model card.

## Converted models on Hugging Face

| Model | Size | Accuracy (fast-decisions dev) | Same answer as fp32 |
|---|---|---|---|
| [souuzaa/GLiNER2.5-Decide-4bit](https://huggingface.co/souuzaa/GLiNER2.5-Decide-4bit) | 350 MB | 63.7% | 97.2% |
| [souuzaa/GLiNER2.5-Decide-8bit](https://huggingface.co/souuzaa/GLiNER2.5-Decide-8bit) | 567 MB | 63.8% | 99.6% |
| [souuzaa/GLiNER2.5-Decide-bf16](https://huggingface.co/souuzaa/GLiNER2.5-Decide-bf16) | 973 MB | 63.7% | 99.5% |

The fp32 original scores 63.7% on the same split. Load any of them without cloning this repo:

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("souuzaa/GLiNER2.5-Decide-4bit")
sys.path.insert(0, path)
import gliner2_mlx

model = gliner2_mlx.load(path)
model.classify_text("Can I get that charge refunded?", {"intent": ["refund_request", "order_status", "other"]})
```

## Files

| File | Purpose |
|---|---|
| `gliner2_mlx.py` | MLX model (DeBERTa-v2 encoder + GLiNER2 heads), gliner2-compatible pre/post-processing, `load()` |
| `convert.py` | Download → remap → quantize → write repo (+ optional upload) |
| `verify_parity.py` | Compare against the official PyTorch `gliner2` (needs torch) |
| `evaluate.py` | Accuracy / agreement / latency on the `fastino/fast-decisions` dev split |

## Why MLX on a Mac

The original model runs through PyTorch (`pip install gliner2`). On Apple Silicon this MLX version is the better choice:

- **Built for Apple Silicon.** [MLX](https://github.com/ml-explore/mlx) is Apple's array framework for M-series chips.
  It runs on the GPU through Metal and uses unified memory, so weights are never copied between CPU and GPU memory.
  PyTorch's Mac GPU backend (MPS) is a port of a CUDA-first design, and on this model it is barely faster than the CPU.
- **About 2× faster.** The full encoder runs on the GPU, with fused attention kernels.
- **Much smaller in memory and on disk.** MLX quantizes the encoder to 4 or 8 bits natively. The 4-bit model is 350 MB
  instead of 1.9 GB, and the whole process peaks at about 0.6 GB of RAM instead of about 4.4 GB. That leaves room for
  other apps or models on a 8–16 GB Mac.
- **Same answers.** In fp32 the port matches PyTorch to about 1e-6 in probability. The 4-bit model has the same
  accuracy on the fast-decisions dev split (63.7%; see the model card).
- **Lighter install.** It needs only `mlx`, `numpy` and `tokenizers`: no torch or transformers (several GB of
  dependencies), so the environment is simpler to ship.

Measured on an Apple M4 (16 GB), `classify_text` on 102 fast-decisions rows at batch size 1, after warm-up:

| Runtime | Median latency | p90 latency | Peak process RAM |
|---|---|---|---|
| PyTorch `gliner2`, CPU (fp32) | 220 ms | 332 ms | 4.39 GB |
| PyTorch `gliner2`, MPS GPU (fp32) | 215 ms | 295 ms | 4.39 GB |
| PyTorch `gliner2`, MPS GPU (fp16) | 180 ms | 247 ms | 4.39 GB |
| **MLX bf16** | **105 ms** | **151 ms** | **1.23 GB** |
| **MLX 4-bit** | **99 ms** | **135 ms** | **0.63 GB** |

Peak RAM is the process's max resident set size, which includes the framework itself. Numbers vary by chip.

Use the PyTorch version instead if you're on Linux or Windows, need CUDA, want to fine-tune, or need the parts that aren't
ported here (relations, JSON structures, long-document chunking).

## Convert

```bash
uv venv -p 3.12 && uv pip install -r requirements.txt
python convert.py                         # -> mlx_models/GLiNER2.5-Decide-4bit
python convert.py --q-bits 8              # -> mlx_models/GLiNER2.5-Decide-8bit
python convert.py --no-quantize --dtype bfloat16
```

Options: `--hf-path` (source repo or dir), `--mlx-path`, `--q-bits`, `--q-group-size`, `--dtype` (unquantized weights),
`--quality quality.md` (appends an evaluation table to the model card), `--upload-repo <org/name>`.

Quantization covers the encoder's linear layers and word embeddings; relative-position embeddings and all GLiNER2 heads
stay in `--dtype`.

## Verify

```bash
uv pip install "gliner2[local]"                                  # torch, only for the parity check
python verify_parity.py                                          # original weights in fp32 vs PyTorch
python verify_parity.py --mlx mlx_models/GLiNER2.5-Decide-4bit
python evaluate.py mlx_models/GLiNER2.5-Decide-* --markdown quality.md
python convert.py --q-bits 4 --quality quality.md                # bake the table into the model card
```

## Use

```python
import gliner2_mlx

model = gliner2_mlx.load("mlx_models/GLiNER2.5-Decide-4bit")
model.classify_text("Can I get that charge refunded?", {"intent": ["refund_request", "order_status", "other"]})
model.predict("Can I get that charge refunded?", {"intent": ["refund_request", "order_status", "other"]})
model.extract_entities("Tim Cook announced the iPhone in Cupertino.", ["person", "product", "location"])
```

Not ported: relation extraction, JSON-structure extraction, the `*_long` chunking helpers, and non-span
(`boundary`) GLiNER2 architectures.
