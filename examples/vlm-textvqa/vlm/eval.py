"""TextVQA-val evaluation: accuracy + side-by-side markdown grid.

Loads the base Qwen2.5-VL model (with vision tower this time, needed
for unseen inference images), optionally applies the LoRA adapter, and
runs greedy generation on every row of the eval Lance dataset.

Outputs:

  * ``accuracy.json``: scalar TextVQA accuracy (see ``_score_one``)
  * ``predictions.jsonl``: one row per example with question, GT
    answers, and generated answer
  * ``side_by_side.md``: K examples with image, question, base vs
    tuned answer, GT
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import time
from pathlib import Path

import lancedb
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

LOG = logging.getLogger("vlm.eval")

_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _split_db(db: str) -> tuple[str, str]:
    p = Path(db)
    name = p.name[:-len(".lance")] if p.name.endswith(".lance") else p.name
    return str(p.parent), name


# ---------------------------------------------------------------------------
# Scoring (official TextVQA: min(matches/3, 1) with text normalisation)
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[\.\,\!\?\;\:\(\)\"]+")
_WHITE_RE = re.compile(r"\s+")
_ARTICLES = {"a", "an", "the"}


def _normalise(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    s = _WHITE_RE.sub(" ", s)
    s = " ".join(t for t in s.split() if t not in _ARTICLES)
    return s


def _score_one(pred: str, gts: list[str]) -> float:
    if not pred:
        return 0.0
    p = _normalise(pred)
    matches = sum(1 for g in gts if _normalise(g) == p)
    return min(matches / 3.0, 1.0)


# ---------------------------------------------------------------------------
# Model + generation
# ---------------------------------------------------------------------------

def _load_model(adapter_dir: str | None, load_4bit: bool = False):
    LOG.info("loading base model %s%s", _QWEN_MODEL_ID,
             " (4-bit NF4)" if load_4bit else "")
    kwargs = dict(attn_implementation="sdpa")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "cuda:0"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(_QWEN_MODEL_ID, **kwargs)
    if adapter_dir:
        LOG.info("loading LoRA adapter from %s", adapter_dir)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        # merge_and_unload can't fold LoRA into 4-bit weights; keep the
        # adapter active for generation in that case.
        if not load_4bit:
            model = model.merge_and_unload()
    model.eval()
    processor = AutoProcessor.from_pretrained(_QWEN_MODEL_ID)
    return model, processor


_TEXTVQA_HINT = "Answer the question using a single word or short phrase, no explanation."


@torch.no_grad()
def _generate(model, processor, image: Image.Image, question: str,
              max_new_tokens: int = 16) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": f"{question}\n\n{_TEXTVQA_HINT}"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def _iter_eval_rows(tbl, limit: int | None):
    cols = ["id", "question_id", "image", "question", "answer", "answers"]
    n = limit if limit is not None else tbl.count_rows()
    table = tbl.search().select(cols).limit(n).to_arrow()
    for i in range(table.num_rows):
        yield {
            "id":          table.column("id")[i].as_py(),
            "question_id": table.column("question_id")[i].as_py(),
            "image":       table.column("image")[i].as_py(),
            "question":    table.column("question")[i].as_py(),
            "answer":      table.column("answer")[i].as_py(),
            "answers":     table.column("answers")[i].as_py(),
        }


def _evaluate(model_label: str, model, processor, tbl, limit, out_dir: Path) -> dict:
    LOG.info("evaluating %s (limit=%s)", model_label, limit)
    preds_path = out_dir / f"predictions_{model_label}.jsonl"
    pf = preds_path.open("w")
    scores: list[float] = []
    t0 = time.time()
    for i, row in enumerate(_iter_eval_rows(tbl, limit)):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        pred = _generate(model, processor, img, row["question"])
        score = _score_one(pred, row["answers"])
        scores.append(score)
        rec = {
            "question_id": row["question_id"],
            "question":    row["question"],
            "gt_answer":   row["answer"],
            "gt_answers":  row["answers"],
            "prediction":  pred,
            "score":       score,
        }
        pf.write(json.dumps(rec) + "\n")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            LOG.info("  %s: %d done  acc=%.3f  ips=%.2f",
                     model_label, i + 1, sum(scores) / len(scores), (i + 1) / elapsed)
    pf.close()
    acc = sum(scores) / max(len(scores), 1)
    summary = {"model": model_label, "n": len(scores), "accuracy": acc,
               "wall_s": time.time() - t0}
    LOG.info("%s FINAL: n=%d acc=%.4f time=%.1fs",
             model_label, summary["n"], acc, summary["wall_s"])
    return summary


# ---------------------------------------------------------------------------
# Side-by-side markdown table
# ---------------------------------------------------------------------------

def _b64_thumb(image_bytes: bytes, size: int = 192) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_side_by_side(
    tbl, base_preds: list[dict], tuned_preds: list[dict],
    out_md: Path, k: int = 8,
):
    """Render top-K rows where base lost but tuned won, then K random rows."""
    base_by_id  = {r["question_id"]: r for r in base_preds}
    tuned_by_id = {r["question_id"]: r for r in tuned_preds}

    common_ids = [qid for qid in base_by_id if qid in tuned_by_id]
    wins = [qid for qid in common_ids
            if tuned_by_id[qid]["score"] > base_by_id[qid]["score"]]

    LOG.info("side-by-side: %d common, %d tuned-wins; rendering top %d wins + %d random",
             len(common_ids), len(wins), k, k)

    # Map question_id -> image bytes (need to re-read those rows)
    needed = set(wins[:k] + common_ids[:k])
    img_by_id: dict[str, bytes] = {}
    table = tbl.search().select(["question_id", "image"]).limit(tbl.count_rows()).to_arrow()
    for i in range(table.num_rows):
        qid = table.column("question_id")[i].as_py()
        if qid in needed:
            img_by_id[qid] = table.column("image")[i].as_py()

    lines: list[str] = [
        "# TextVQA — base vs LoRA-tuned, side by side\n",
        f"_Base accuracy: {sum(r['score'] for r in base_preds) / len(base_preds):.3f}_",
        f"_Tuned accuracy: {sum(r['score'] for r in tuned_preds) / len(tuned_preds):.3f}_\n",
        "## Tuned-wins (where tuned beat base)\n",
        "| Image | Question | Base | Tuned | GT |",
        "|---|---|---|---|---|",
    ]
    for qid in wins[:k]:
        b = base_by_id[qid]; t = tuned_by_id[qid]
        img_b64 = _b64_thumb(img_by_id[qid])
        gts = b["gt_answers"][:5]
        lines.append(
            f"| <img src=\"data:image/jpeg;base64,{img_b64}\" width=192/> "
            f"| {b['question']} | {b['prediction']} | **{t['prediction']}** "
            f"| {', '.join(gts)} |"
        )

    lines.append("\n## Random samples\n")
    lines.append("| Image | Question | Base | Tuned | GT |")
    lines.append("|---|---|---|---|---|")
    for qid in common_ids[:k]:
        b = base_by_id[qid]; t = tuned_by_id[qid]
        img_b64 = _b64_thumb(img_by_id[qid])
        gts = b["gt_answers"][:5]
        lines.append(
            f"| <img src=\"data:image/jpeg;base64,{img_b64}\" width=192/> "
            f"| {b['question']} | {b['prediction']} | {t['prediction']} "
            f"| {', '.join(gts)} |"
        )

    out_md.write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", out_md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--db",        default="data/textvqa_val.lance",
                   help="eval Lance dataset")
    p.add_argument("--adapter",   default=None,
                   help="path to LoRA adapter (omit to eval base model only)")
    p.add_argument("--out",       default="eval_outputs",
                   help="output dir")
    p.add_argument("--limit",     type=int, default=200,
                   help="cap rows (default 200; full val = 5000)")
    p.add_argument("--mode",      default="both",
                   choices=["base", "tuned", "both"])
    p.add_argument("--side-by-side-k", type=int, default=8)
    p.add_argument("--load-4bit", action="store_true",
                   help="load Qwen in 4-bit (NF4) for low-VRAM eval on a Colab T4")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    uri, table_name = _split_db(args.db)
    tbl = lancedb.connect(uri).open_table(table_name)
    LOG.info("eval table: %s/%s rows=%d", uri, table_name, tbl.count_rows())

    results: dict[str, dict] = {}
    base_preds: list[dict] = []
    tuned_preds: list[dict] = []

    if args.mode in ("base", "both"):
        model, proc = _load_model(adapter_dir=None, load_4bit=args.load_4bit)
        results["base"] = _evaluate("base", model, proc, tbl, args.limit, out_dir)
        base_preds = [json.loads(l) for l in (out_dir / "predictions_base.jsonl").open()]
        del model; torch.cuda.empty_cache()

    if args.mode in ("tuned", "both") and args.adapter:
        model, proc = _load_model(adapter_dir=args.adapter, load_4bit=args.load_4bit)
        results["tuned"] = _evaluate("tuned", model, proc, tbl, args.limit, out_dir)
        tuned_preds = [json.loads(l) for l in (out_dir / "predictions_tuned.jsonl").open()]
        del model; torch.cuda.empty_cache()

    with (out_dir / "accuracy.json").open("w") as f:
        json.dump(results, f, indent=2)
    LOG.info("accuracy: %s", results)

    if base_preds and tuned_preds:
        _build_side_by_side(
            tbl, base_preds, tuned_preds, out_dir / "side_by_side.md",
            k=args.side_by_side_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
