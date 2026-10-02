"""MLX port of GLiNER2 span checkpoints (DeBERTa-v3 encoder + GLiNER2 heads).

Torch-free. Needs only ``mlx``, ``numpy`` and ``tokenizers`` (plus
``huggingface_hub`` to load by repo id). Loads the converted MLX checkpoint
(bf16/fp16 or quantized) and also the original ``fastino/*`` checkpoints as-is.

    import gliner2_mlx
    model = gliner2_mlx.load("mlx-community/GLiNER2.5-Decide-4bit")
    model.classify_text(text, {"intent": ["refund", "cancel", "other"]})
    model.extract_entities(text, ["person", "company"])

Pre/post-processing mirrors ``gliner2`` 2.0.0 (``SchemaTransformer`` and
``ExtractorRuntimeMixin``) for classification and entity extraction.
Relation and JSON-structure extraction are not ported.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SEP_STRUCT, SEP_TEXT = "[SEP_STRUCT]", "[SEP_TEXT]"
P_TOKEN, E_TOKEN, L_TOKEN = "[P]", "[E]", "[L]"
EXAMPLE_TOKEN, OUTPUT_TOKEN, DESC_TOKEN = "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"


# --------------------------------------------------------------------------- encoder
# transformers.models.deberta_v2 (DeBERTa-v3: relative attention, log buckets,
# share_att_key, c2p + p2c, LayerNorm on relative embeddings, no conv layer).


def _log_bucket_position(rel: np.ndarray, bucket_size: int, max_position: int) -> np.ndarray:
    """make_log_bucket_position, computed in float32 like the torch original."""
    sign = np.sign(rel)
    mid = bucket_size // 2
    abs_pos = np.where((rel < mid) & (rel > -mid), mid - 1, np.abs(rel))
    log_pos = (
        np.ceil(
            np.log(abs_pos.astype(np.float32) / np.float32(mid))
            / np.log(np.float32((max_position - 1) / mid))
            * np.float32(mid - 1)
        )
        + mid
    )
    return np.where(abs_pos <= mid, rel, log_pos * sign).astype(np.int64)


@lru_cache(maxsize=16)
def _rel_indices(L: int, buckets: int, max_position: int) -> Tuple[mx.array, mx.array]:
    """Gather indices for c2p and p2c (both (L, L)); the caller transposes the p2c term."""
    pos = np.arange(L)
    rel = _log_bucket_position(pos[:, None] - pos[None, :], buckets, max_position)
    c2p = np.clip(rel + buckets, 0, 2 * buckets - 1)
    p2c = np.clip(-rel + buckets, 0, 2 * buckets - 1)
    return mx.array(c2p, dtype=mx.int32), mx.array(p2c, dtype=mx.int32)


class DisentangledSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query_proj = nn.Linear(hidden, hidden)
        self.key_proj = nn.Linear(hidden, hidden)
        self.value_proj = nn.Linear(hidden, hidden)

    def _split(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        return x.reshape(B, L, self.heads, D // self.heads).transpose(0, 2, 1, 3)

    def __call__(self, x, key_bias, rel_emb, c2p_idx, p2c_idx):
        q = self._split(self.query_proj(x))
        k = self._split(self.key_proj(x))
        v = self._split(self.value_proj(x))
        pos_q = self._split(self.query_proj(rel_emb))  # (1, H, 2*span, hd), share_att_key
        pos_k = self._split(self.key_proj(rel_emb))
        B, H, L, hd = q.shape
        scale = 1.0 / math.sqrt(hd * 3)  # 1 + c2p + p2c

        shape = (B, H, L, L)
        c2p = mx.take_along_axis(q @ pos_k.transpose(0, 1, 3, 2), mx.broadcast_to(c2p_idx, shape), axis=-1)
        p2c = mx.take_along_axis(k @ pos_q.transpose(0, 1, 3, 2), mx.broadcast_to(p2c_idx, shape), axis=-1)
        bias = (c2p + p2c.transpose(0, 1, 3, 2)) * scale + key_bias
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias.astype(q.dtype))
        return out.transpose(0, 2, 1, 3).reshape(B, L, H * hd)


class _Dense(nn.Module):
    """dense -> (+ residual) -> LayerNorm, as in DebertaV2SelfOutput / DebertaV2Output."""

    def __init__(self, in_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def __call__(self, x, residual):
        return self.LayerNorm(self.dense(x) + residual)


class _Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, eps: float):
        super().__init__()
        setattr(self, "self", DisentangledSelfAttention(hidden, heads))
        self.output = _Dense(hidden, hidden, eps)

    def __call__(self, x, *args):
        return self.output(getattr(self, "self")(x, *args), x)


class _Intermediate(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.dense = nn.Linear(hidden, inner)

    def __call__(self, x):
        return nn.gelu(self.dense(x))


class DebertaV2Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, inner: int, eps: float):
        super().__init__()
        self.attention = _Attention(hidden, heads, eps)
        self.intermediate = _Intermediate(hidden, inner)
        self.output = _Dense(inner, hidden, eps)

    def __call__(self, x, *args):
        a = self.attention(x, *args)
        return self.output(self.intermediate(a), a)


class _Embeddings(nn.Module):
    def __init__(self, vocab: int, hidden: int, eps: float):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def __call__(self, ids, mask):
        return self.LayerNorm(self.word_embeddings(ids)) * mask[..., None]


class _Encoder(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        hidden, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.layer = [
            DebertaV2Layer(hidden, cfg["num_attention_heads"], cfg["intermediate_size"], eps)
            for _ in range(cfg["num_hidden_layers"])
        ]
        self.rel_embeddings = nn.Embedding(cfg["position_buckets"] * 2, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)


class DebertaV2Model(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        unsupported = {
            "relative_attention": True, "share_att_key": True, "position_biased_input": False,
            "norm_rel_ebd": "layer_norm", "type_vocab_size": 0,
        }
        for key, expected in unsupported.items():
            if cfg.get(key, expected) != expected:
                raise NotImplementedError(f"encoder config {key}={cfg[key]!r} is not supported")
        if cfg.get("conv_kernel_size", 0) or sorted(cfg.get("pos_att_type", [])) != ["c2p", "p2c"]:
            raise NotImplementedError("only DeBERTa-v3 style encoders are supported")
        self.position_buckets = cfg["position_buckets"]
        max_rel = cfg.get("max_relative_positions", -1)
        self.max_relative_positions = max_rel if max_rel > 0 else cfg["max_position_embeddings"]
        self.embeddings = _Embeddings(cfg["vocab_size"], cfg["hidden_size"], cfg["layer_norm_eps"])
        self.encoder = _Encoder(cfg)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        dtype = self.encoder.LayerNorm.weight.dtype
        mask = attention_mask.astype(dtype)
        x = self.embeddings(input_ids, mask)
        # Mask keys only; padded query rows stay finite and are discarded later.
        key_bias = mx.where(attention_mask[:, None, None, :] > 0, 0.0, -mx.inf).astype(mx.float32)
        rel_emb = self.encoder.LayerNorm(self.encoder.rel_embeddings.weight)[None]
        c2p, p2c = _rel_indices(input_ids.shape[1], self.position_buckets, self.max_relative_positions)
        for layer in self.encoder.layer:
            x = layer(x, key_bias, rel_emb, c2p, p2c)
        return x


# --------------------------------------------------------------------------- heads


class MLP(nn.Module):
    """Linear -> ReLU -> Linear (gliner2 create_mlp / create_projection_layer)."""

    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class SpanMarkerV0(nn.Module):
    def __init__(self, hidden: int, max_width: int):
        super().__init__()
        self.max_width = max_width
        self.project_start = MLP(hidden, hidden * 4, hidden)
        self.project_end = MLP(hidden, hidden * 4, hidden)
        self.out_project = MLP(hidden * 2, hidden * 4, hidden)

    def __call__(self, h: mx.array) -> mx.array:
        """h: (L, D) word embeddings -> (L, max_width, D) span representations."""
        L = h.shape[0]
        starts = np.repeat(np.arange(L), self.max_width).reshape(L, self.max_width)
        ends = starts + np.arange(self.max_width)[None]
        valid = ends < L
        starts, ends = mx.array(np.where(valid, starts, 0)), mx.array(np.where(valid, ends, 0))
        cat = mx.concatenate([self.project_start(h)[starts], self.project_end(h)[ends]], axis=-1)
        return self.out_project(nn.relu(cat))


class _SpanRep(nn.Module):
    def __init__(self, hidden: int, max_width: int):
        super().__init__()
        self.span_rep_layer = SpanMarkerV0(hidden, max_width)


class CountLSTM(nn.Module):
    def __init__(self, hidden: int, max_count: int = 20):
        super().__init__()
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden)
        self.gru = GRU(hidden)
        self.projector = MLP(hidden * 2, hidden * 4, hidden)

    def __call__(self, pc_emb: mx.array, count: int) -> mx.array:
        """pc_emb: (M, D) field embeddings -> (count, M, D)."""
        count = min(count, self.max_count)
        pos = self.pos_embedding(mx.arange(count))
        out = self.gru(mx.broadcast_to(pos[:, None], (count, *pc_emb.shape)), pc_emb)
        return self.projector(mx.concatenate([out, mx.broadcast_to(pc_emb[None], out.shape)], axis=-1))


class GRU(nn.Module):
    """Single-layer torch.nn.GRU semantics (gate order r, z, n)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.weight_ih_l0 = mx.zeros((3 * hidden, hidden))
        self.weight_hh_l0 = mx.zeros((3 * hidden, hidden))
        self.bias_ih_l0 = mx.zeros((3 * hidden,))
        self.bias_hh_l0 = mx.zeros((3 * hidden,))

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        outputs = []
        for t in range(x.shape[0]):
            i_r, i_z, i_n = mx.split(x[t] @ self.weight_ih_l0.T + self.bias_ih_l0, 3, axis=-1)
            h_r, h_z, h_n = mx.split(h @ self.weight_hh_l0.T + self.bias_hh_l0, 3, axis=-1)
            r = mx.sigmoid(i_r + h_r)
            z = mx.sigmoid(i_z + h_z)
            n = mx.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outputs.append(h)
        return mx.stack(outputs)


class GLiNER2Model(nn.Module):
    def __init__(self, config: Dict[str, Any], encoder_config: Dict[str, Any]):
        super().__init__()
        if config.get("architecture", "span") != "span":
            raise NotImplementedError(f"architecture {config['architecture']!r} is not supported")
        if config.get("counting_layer", "count_lstm") != "count_lstm":
            raise NotImplementedError(f"counting_layer {config['counting_layer']!r} is not supported")
        hidden = encoder_config["hidden_size"]
        self.max_width = config.get("max_width", 8)
        self.encoder = DebertaV2Model(encoder_config)
        self.span_rep = _SpanRep(hidden, self.max_width)
        self.classifier = MLP(hidden, hidden * 2, 1)
        self.count_pred = MLP(hidden, hidden * 2, 20)
        self.count_embed = CountLSTM(hidden)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map PyTorch nn.Sequential indices to named MLP layers (idempotent)."""
        out = {}
        for k, v in weights.items():
            k = re.sub(r"^(classifier|count_pred|count_embed\.projector)\.0\.", r"\1.fc1.", k)
            k = re.sub(r"^(classifier|count_pred|count_embed\.projector)\.2\.", r"\1.fc2.", k)
            k = re.sub(r"(project_start|project_end|out_project)\.0\.", r"\1.fc1.", k)
            k = re.sub(r"(project_start|project_end|out_project)\.3\.", r"\1.fc2.", k)
            out[k] = v
        return out


def quantization_predicate(path: str, module: nn.Module) -> bool:
    """Quantize encoder Linear/Embedding layers; keep relative embeddings and heads in float."""
    return (
        path.startswith("encoder.")
        and "rel_embeddings" not in path
        and hasattr(module, "to_quantized")
        and module.weight.shape[-1] % 64 == 0
    )


# --------------------------------------------------------------------------- preprocessing
# gliner2.processor.SchemaTransformer (inference mode).


class WhitespaceTokenSplitter:
    _PATTERN = re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text: str, lower: bool = True):
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()


class CharLevelSplitter:
    _PATTERN = re.compile(r"[A-Za-z0-9@._\-+]+|\S")

    def __call__(self, text: str, lower: bool = True):
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()


WORD_SPLITTERS = {"whitespace": WhitespaceTokenSplitter, "char": CharLevelSplitter}


@dataclass
class _Task:
    kind: str  # "entities" | "classifications"
    tokens: List[str]
    config: Dict[str, Any]


@dataclass
class _Record:
    text: str
    input_ids: List[int]
    tasks: List[_Task]
    marker_positions: List[List[int]]
    word_positions: List[int]
    start_map: List[int]
    end_map: List[int]


def _transform_schema(parent, fields, child_prefix, prompt=None, examples=None, label_descriptions=None, mode="both"):
    prompt_str = f"{parent}: {prompt}" if prompt else parent
    if mode in ("descriptions", "both") and label_descriptions:
        for label, desc in label_descriptions.items():
            if label in fields:
                prompt_str += f" {DESC_TOKEN} {label}: {desc}"
    if mode in ("few_shot", "both") and examples:
        for inp, out in examples:
            if out in fields:
                out_str = out if isinstance(out, str) else ", ".join(out)
                prompt_str += f" {EXAMPLE_TOKEN} {inp} {OUTPUT_TOKEN} {out_str}"
    tokens = ["(", P_TOKEN, prompt_str, "("]
    for name in fields:
        tokens.extend([child_prefix, name])
    tokens.extend([")", ")"])
    return tokens


def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Accept a gliner2 ``Schema`` (``.build()``) or a schema dict; keep entities + classifications."""
    entity_meta = getattr(schema, "_entity_metadata", None)
    if hasattr(schema, "build"):
        schema = schema.build()
    for key in ("json_structures", "relations"):
        if schema.get(key):
            raise NotImplementedError(f"{key} extraction is not supported by gliner2_mlx")
    entities = schema.get("entities") or {}
    if isinstance(entities, (list, tuple)):
        entities = {e: "" for e in entities}
    out = {
        "entities": list(entities),
        "entity_descriptions": dict(schema.get("entity_descriptions") or {}),
        "entity_metadata": dict(entity_meta or schema.get("entity_metadata") or {}),
        "classifications": [dict(c) for c in schema.get("classifications") or []],
    }
    return out


def classification_schema(tasks: Dict[str, Any]) -> Dict[str, Any]:
    """gliner2 ``_classification_schema`` + ``Schema.classification``."""
    classifications = []
    for name, config in tasks.items():
        cfg = dict(config) if isinstance(config, dict) and "labels" in config else {"labels": config}
        labels = cfg.pop("labels")
        item = {
            "task": name,
            "labels": list(labels.keys()) if isinstance(labels, dict) else list(labels),
            "multi_label": cfg.pop("multi_label", False),
            "cls_threshold": cfg.pop("cls_threshold", 0.5),
            **cfg,
        }
        if isinstance(labels, dict):
            item["label_descriptions"] = labels
        classifications.append(item)
    return {"classifications": classifications}


def entity_schema(entity_types: Union[str, Sequence[str], Dict[str, Any]], **defaults) -> Dict[str, Any]:
    """gliner2 ``Schema.entities``: names, ``{name: description}`` or ``{name: {description, dtype, threshold}}``."""
    if isinstance(entity_types, str):
        entity_types = [entity_types]
    if not isinstance(entity_types, dict):
        entity_types = {name: {} for name in entity_types}
    names, descriptions, metadata = [], {}, {}
    for name, config in entity_types.items():
        config = {"description": config} if isinstance(config, str) else dict(config or {})
        names.append(name)
        if config.get("description"):
            descriptions[name] = config["description"]
        metadata[name] = {
            "dtype": config.get("dtype", defaults.get("dtype", "list")),
            "threshold": config.get("threshold", defaults.get("threshold")),
        }
    return {"entities": names, "entity_descriptions": descriptions, "entity_metadata": metadata}


class Preprocessor:
    def __init__(self, tokenizer, word_splitter: str = "whitespace"):
        self.tokenizer = tokenizer
        self.word_splitter = WORD_SPLITTERS[word_splitter]()
        self._cache: Dict[str, List[str]] = {}

    def tokenize(self, text: str) -> List[str]:
        if text not in self._cache:
            if len(self._cache) > 65536:
                self._cache.clear()
            self._cache[text] = self.tokenizer.encode(text, add_special_tokens=False).tokens
        return self._cache[text]

    def __call__(self, text: str, schema: Dict[str, Any], max_len: Optional[int] = None) -> _Record:
        if text and not text.endswith((".", "!", "?")):
            text = text + "."
        elif not text:
            text = "."

        tasks: List[_Task] = []
        if schema["entities"]:
            descs = schema["entity_descriptions"]
            tasks.append(_Task("entities", _transform_schema(
                "entities", schema["entities"], E_TOKEN, label_descriptions=descs,
                mode="descriptions" if descs else "none"), {}))
        for item in schema["classifications"]:
            tasks.append(_Task("classifications", _transform_schema(
                item["task"], item["labels"], L_TOKEN, prompt=item.get("prompt"),
                examples=item.get("examples", []), label_descriptions=item.get("label_descriptions") or {},
                mode="both"), item))

        words, start_map, end_map = [], [], []
        for tok, start, end in self.word_splitter(text, lower=True):
            words.append(tok)
            start_map.append(start)
            end_map.append(end)
        if max_len is not None:
            words, start_map, end_map = words[:max_len], start_map[:max_len], end_map[:max_len]

        subwords: List[str] = []
        marker_positions: List[List[int]] = []
        for i, task in enumerate(tasks):
            if i:
                subwords.extend(self.tokenize(SEP_STRUCT))
            markers = {1} | set(range(4, len(task.tokens) - 2, 2))
            positions = []
            for j, token in enumerate(task.tokens):
                if j in markers:
                    positions.append(len(subwords))
                subwords.extend(self.tokenize(token))
            marker_positions.append(positions)
        subwords.extend(self.tokenize(SEP_TEXT))
        word_positions = []
        for word in words:
            word_positions.append(len(subwords))
            subwords.extend(self.tokenize(word))
        word_positions = [min(p, len(subwords) - 1) for p in word_positions]
        unk = self.tokenizer.token_to_id("[UNK]")
        ids = [self.tokenizer.token_to_id(t) for t in subwords]
        ids = [unk if i is None else i for i in ids]
        return _Record(text, ids, tasks, marker_positions, word_positions, start_map, end_map)


# --------------------------------------------------------------------------- decoding
# gliner2.inference.runtime.ExtractorRuntimeMixin (classification + entities).


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _class_probs(logits: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    activation = cfg.get("class_act", "auto")
    if activation == "sigmoid" or (activation == "auto" and cfg.get("multi_label", False)):
        return _sigmoid(logits)
    return _softmax(logits)


def _decide(probs: np.ndarray, cfg: Dict[str, Any]):
    labels = cfg["labels"]
    if cfg.get("multi_label", False):
        chosen = [(labels[j], float(probs[j])) for j in range(len(labels)) if probs[j] >= cfg.get("cls_threshold", 0.5)]
        if not chosen:
            best = int(np.argmax(probs))
            chosen = [(labels[best], float(probs[best]))]
        return chosen
    best = int(np.argmax(probs))
    return labels[best], float(probs[best])


def _finalize_spans(spans, dtype="list"):
    """Confidence-first greedy non-overlapping selection (span-architecture default)."""
    kept = []
    for cand in sorted(spans, key=lambda s: s[1], reverse=True):
        if not any(cand[2] < k[3] and k[2] < cand[3] for k in kept):
            kept.append(cand)
    return kept if dtype == "list" else kept[:1]


def _format_entities(spans, include_confidence, include_spans):
    out = []
    for text, conf, start, end in spans:
        item = {"text": text}
        if include_confidence:
            item["confidence"] = conf
        if include_spans:
            item["start"], item["end"] = start, end
        out.append(item if (include_confidence or include_spans) else text)
    return out


def _dedupe(values):
    unique, seen = [], set()
    for v in values:
        if isinstance(v, dict):
            text = v.get("text", "")
            key = (text.lower(), v.get("start"), v.get("end"))
        else:
            text, key = v, v.lower() if v else v
        if text and key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


# --------------------------------------------------------------------------- model


class GLiNER2MLX:
    def __init__(self, model: GLiNER2Model, tokenizer, word_splitter: str = "whitespace"):
        self.model = model
        self.preprocessor = Preprocessor(tokenizer, word_splitter)

    # ---- core
    def _encode(self, records: List[_Record]) -> mx.array:
        L = max(len(r.input_ids) for r in records)
        ids = np.zeros((len(records), L), dtype=np.int32)
        mask = np.zeros((len(records), L), dtype=np.int32)
        for i, r in enumerate(records):
            ids[i, : len(r.input_ids)] = r.input_ids
            mask[i, : len(r.input_ids)] = 1
        return self.model.encoder(mx.array(ids), mx.array(mask))

    def _forward(self, records: List[_Record]) -> List[List[Tuple[_Task, Dict[str, np.ndarray]]]]:
        """Run encoder + heads; returns per record, per task, the raw numpy outputs."""
        hidden = self._encode(records)
        m = self.model
        lazy = []
        for b, rec in enumerate(records):
            h = hidden[b]
            words = h[mx.array(rec.word_positions, dtype=mx.int32)] if rec.word_positions else None
            span_rep = None
            per_task = []
            for task, positions in zip(rec.tasks, rec.marker_positions):
                embs = h[mx.array(positions, dtype=mx.int32)]
                if task.kind == "classifications":
                    per_task.append((task, {"logits": m.classifier(embs[1:])[:, 0]}))
                    continue
                count = int(mx.argmax(m.count_pred(embs[0])).item())
                if count <= 0 or words is None:
                    per_task.append((task, {"count": count}))
                    continue
                if span_rep is None:
                    span_rep = m.span_rep.span_rep_layer(words)
                proj = m.count_embed(embs[1:], count)  # (count, M, D)
                logits = mx.einsum("lkd,bpd->bplk", span_rep, proj)
                per_task.append((task, {"count": count, "scores": mx.sigmoid(logits[0].astype(mx.float32))}))
            lazy.append(per_task)
        mx.eval(lazy)
        return [
            [(t, {k: np.array(v.astype(mx.float32)) if isinstance(v, mx.array) else v for k, v in out.items()})
             for t, out in per_task]
            for per_task in lazy
        ]

    def _decode(self, rec: _Record, outputs, schema, threshold, include_confidence, include_spans):
        result: Dict[str, Any] = {}
        n_words = len(rec.start_map)
        for task, out in outputs:
            if task.kind == "classifications":
                cfg = task.config
                probs = _class_probs(out["logits"] / cfg.get("temperature", 1.0), cfg)
                value = _decide(probs, cfg)
                if isinstance(value, list):
                    result[cfg["task"]] = [{"label": l, "confidence": c} if include_confidence else l for l, c in value]
                else:
                    result[cfg["task"]] = {"label": value[0], "confidence": value[1]} if include_confidence else value[0]
                continue
            if "scores" not in out:
                result["entities"] = {}
                continue
            entities = {}
            for idx, name in enumerate(schema["entities"]):
                meta = schema["entity_metadata"].get(name, {})
                thr = meta.get("threshold")
                thr = threshold if thr is None else float(thr)
                dtype = meta.get("dtype", "list")
                scores = out["scores"][idx]
                spans = []
                for start, width in zip(*np.nonzero(scores >= thr)):
                    end = start + width + 1
                    if end > n_words:
                        continue
                    cs, ce = rec.start_map[start], rec.end_map[end - 1]
                    text = rec.text[cs:ce].strip()
                    if text:
                        spans.append((text, float(scores[start, width]), cs, ce))
                spans = _finalize_spans(spans, dtype)
                formatted = _format_entities(spans, include_confidence, include_spans)
                if dtype == "list":
                    entities[name] = _dedupe(formatted)
                else:
                    entities[name] = formatted[0] if formatted else None
            result["entities"] = entities
        return result

    # ---- public API (mirrors gliner2)
    def batch_extract(
        self,
        texts: List[str],
        schema: Union[Dict[str, Any], List[Dict[str, Any]]],
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        schemas = schema if isinstance(schema, list) else [schema] * len(texts)
        if len(schemas) != len(texts):
            raise ValueError(f"Schema count ({len(schemas)}) != text count ({len(texts)})")
        schemas = [normalize_schema(s) for s in schemas]
        results = []
        for i in range(0, len(texts), batch_size):
            records = [self.preprocessor(t, s, max_len) for t, s in zip(texts[i : i + batch_size], schemas[i : i + batch_size])]
            for rec, outs, s in zip(records, self._forward(records), schemas[i : i + batch_size]):
                results.append(self._decode(rec, outs, s, threshold, include_confidence, include_spans))
        return results

    def extract(self, text: str, schema: Dict[str, Any], **kw) -> Dict[str, Any]:
        return self.batch_extract([text], schema, **kw)[0]

    def classify_text(self, text: str, tasks: Dict[str, Any], **kw) -> Dict[str, Any]:
        return self.extract(text, classification_schema(tasks), **kw)

    def batch_classify_text(self, texts: List[str], tasks: Dict[str, Any], **kw) -> List[Dict[str, Any]]:
        return self.batch_extract(texts, classification_schema(tasks), **kw)

    def extract_entities(self, text: str, entity_types, **kw) -> Dict[str, Any]:
        return self.extract(text, entity_schema(entity_types), **kw)

    def batch_extract_entities(self, texts: List[str], entity_types, **kw) -> List[Dict[str, Any]]:
        return self.batch_extract(texts, entity_schema(entity_types), **kw)

    def predict(self, text: str, tasks: Dict[str, Any], max_len: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Full probability distribution for every classification task: {task: {label: prob}}."""
        schema = normalize_schema(classification_schema(tasks))
        rec = self.preprocessor(text, schema, max_len)
        out = {}
        for task, raw in self._forward([rec])[0]:
            cfg = task.config
            probs = _class_probs(raw["logits"] / cfg.get("temperature", 1.0), cfg)
            out[cfg["task"]] = dict(zip(cfg["labels"], probs.tolist()))
        return out


# --------------------------------------------------------------------------- loading


def _read_configs(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = json.loads((path / "config.json").read_text())
    encoder_config = config.get("encoder_config")
    if encoder_config is None:
        encoder_config = json.loads((path / "encoder_config" / "config.json").read_text())
    return config, encoder_config


def load_weights(path: Path) -> Dict[str, mx.array]:
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    return GLiNER2Model.sanitize(weights)


def build_model(config, encoder_config, weights, dtype=None) -> GLiNER2Model:
    model = GLiNER2Model(config, encoder_config)
    q = config.get("quantization")
    if q:
        nn.quantize(
            model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"),
            class_predicate=lambda p, m: f"{p}.scales" in weights,
        )
    model.load_weights(list(weights.items()), strict=True)
    if dtype is not None:
        model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def load(path: Union[str, Path], dtype: Optional[mx.Dtype] = None, word_splitter: str = "whitespace") -> GLiNER2MLX:
    """Load a GLiNER2 checkpoint (local dir or HF repo id), MLX-converted or original.

    ``dtype`` casts the float parameters (e.g. ``mx.float32`` for closest parity);
    by default the stored dtype is used (original fastino checkpoints are float32).
    """
    from tokenizers import Tokenizer

    path = Path(path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(path), allow_patterns=["*.json", "*.safetensors", "*.py", "*.md"]))
    config, encoder_config = _read_configs(path)
    model = build_model(config, encoder_config, load_weights(path), dtype)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    return GLiNER2MLX(model, tokenizer, word_splitter)
