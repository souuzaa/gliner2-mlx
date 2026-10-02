#!/usr/bin/env python3
"""Parity check: gliner2_mlx vs the official PyTorch gliner2 implementation.

Compares tokenization (input ids, marker and word positions), classifier
probabilities and entity outputs on the examples from the model card.
Requires ``pip install "gliner2[local]"`` (torch) in addition to the MLX deps.

    python verify_parity.py                                   # original checkpoint in fp32
    python verify_parity.py --mlx mlx_models/GLiNER2.5-Decide-4bit
"""

from __future__ import annotations

import argparse
import contextlib
import io

import mlx.core as mx
import numpy as np

import gliner2_mlx

CLASSIFY = [
    ("My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?",
     {"intent": ["order_status", "refund_request", "cancel_subscription", "update_payment", "login_problem",
                 "shipping_delay", "bug_report", "speak_to_human", "other"]}),
    ("The transfer I sent this morning is still pending, and I think I used the wrong sort code. Can you stop it and add Emily as the beneficiary instead?",
     {"intent": ["transfer_pending", "transfer_cancel", "beneficiary_add", "card_lost", "balance_inquiry",
                 "fraud_report", "mortgage_application", "fee_explanation"]}),
    ("Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
     {"sentiment": ["positive", "negative", "mixed", "neutral"],
      "aspects": {"labels": ["battery", "keyboard", "screen", "camera", "price", "support"],
                  "multi_label": True, "cls_threshold": 0.4}}),
    ("INVOICE 1842\nBill to: Northstar QA\nAmount due: 2,400 USD\nDue: 30 April 2026\nWire instructions are on page 2.",
     {"document_type": ["invoice", "receipt", "contract", "resume", "support_email", "meeting_notes"]}),
    ("From: compliance@group.example\nSubject: Protocol update — action required today\n\nPlease confirm the new retention rule is applied before Friday's audit.",
     {"intent": ["fyi", "request", "approval", "complaint", "newsletter", "security_alert"],
      "urgency": ["low", "normal", "high", "critical"],
      "route": ["support", "billing", "legal", "security", "finance", "archive"]}),
    ("The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory ratified it.",
     {"answer": {"labels": ["yes", "no"], "prompt": "Did the treaty enter into force in 1992?"}}),
    ("Please reset the card PIN. The new one never arrived and the old one is locked after three tries.",
     {"intent": {"labels": {"card_pin_change": "The customer wants a new PIN or the current PIN replaced",
                            "card_lost": "The physical card is missing",
                            "balance_inquiry": "The customer wants the current balance"}}}),
    ("I finished it in two nights. The ending is earned, the middle drags, and I would still hand it to a friend",
     {"rating": [str(i) for i in range(11)]}),
    ("", {"label": ["spam", "ham"]}),
]

ENTITIES = [
    ("Tim Cook announced the new iPhone at Apple Park in Cupertino on Tuesday.",
     ["person", "company", "product", "location", "date"]),
    ("Dr. Maria Silva joined Fastino AI in San Francisco as head of research in March 2025.",
     {"person": "Full name of a person", "organization": "Company or institution", "location": "City or place"}),
]


def torch_probs(model, text, tasks):
    """Classifier probabilities from the official implementation (same activation rules)."""
    import torch

    schema = model._classification_schema(tasks)
    batch = model.processor.collate_fn_inference([(text, schema)])
    with torch.no_grad():
        hidden = model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state[0]
        out = {}
        for cfg, positions in zip(schema.build()["classifications"], batch.schema_special_indices[0]):
            logits = model.classifier(hidden[positions[1:]]).squeeze(-1).numpy()
            out[cfg["task"]] = dict(zip(cfg["labels"], gliner2_mlx._class_probs(logits, cfg).tolist()))
    return batch, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", default="fastino/GLiNER2.5-Decide")
    p.add_argument("--mlx", help="converted MLX dir (default: the original checkpoint, run in fp32)")
    args = p.parse_args()

    from gliner2 import AutoExtractor

    with contextlib.redirect_stdout(io.StringIO()):
        ref = AutoExtractor.from_pretrained(args.hf_path)
    ref.eval()
    mlx = gliner2_mlx.load(args.mlx) if args.mlx else gliner2_mlx.load(args.hf_path, dtype=mx.float32)

    ids_ok, agree, deltas = 0, 0, []
    for text, tasks in CLASSIFY:
        batch, tp = torch_probs(ref, text, tasks)
        rec = mlx.preprocessor(text, gliner2_mlx.normalize_schema(gliner2_mlx.classification_schema(tasks)))
        ids_ok += (
            batch.input_ids[0].tolist() == rec.input_ids
            and batch.schema_special_indices[0] == rec.marker_positions
            and batch.text_word_indices[0].tolist()[: len(rec.word_positions)] == rec.word_positions
        )
        mp = mlx.predict(text, tasks)
        for task in tp:
            agree += max(tp[task], key=tp[task].get) == max(mp[task], key=mp[task].get)
            deltas.append(max(abs(tp[task][l] - mp[task][l]) for l in tp[task]))
        same_output = ref.classify_text(text, tasks) == mlx.classify_text(text, tasks)
        print(f"{'OK ' if same_output else 'DIFF'} {mlx.classify_text(text, tasks)}")
    n_tasks = len(deltas)

    ent_same = 0
    for text, types in ENTITIES:
        r = ref.extract_entities(text, types, include_confidence=True)
        m = mlx.extract_entities(text, types, include_confidence=True)
        strip = lambda d: {k: [e["text"] for e in v] for k, v in d["entities"].items()}
        ent_same += strip(r) == strip(m)
        conf = [abs(a["confidence"] - b["confidence"]) for k in r["entities"]
                for a, b in zip(r["entities"][k], m["entities"].get(k, []))]
        print(f"{'OK ' if strip(r) == strip(m) else 'DIFF'} {strip(m)}  max Δconf={max(conf, default=0):.2e}")

    print(f"\ninput ids identical: {ids_ok}/{len(CLASSIFY)}")
    print(f"top answer agrees:   {agree}/{n_tasks} classification tasks, max abs Δprob {max(deltas):.2e}")
    print(f"entities identical:  {ent_same}/{len(ENTITIES)}")
    if ids_ok != len(CLASSIFY) or agree != n_tasks or ent_same != len(ENTITIES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
