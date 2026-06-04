"""Canonical TextVQA curation-slice definitions.

One source of truth shared by the empirical slice picker
(``vlm/slice_experiment.py``) and the Colab bake (``vlm/colab_prepare.py``).

A slice is a predicate over a row's ``question`` + ``ocr_tokens``.  We keep
three candidates (see COLAB_BUILD_PROMPT.md step 1):

  * ``scene_text``  — questions that read specific text out of the image
                      (what number/time/brand/name/letter/word, how much,
                      how many).  These are the hardest for the base model
                      and where LoRA helps most.
  * ``text_dense``  — images with lots of OCR text (``ocr_token_count`` at or
                      above ``TEXT_DENSE_OCR_THRESHOLD``).
  * ``random``      — control: keep everything.
"""
from __future__ import annotations

import re

# Questions that ask the model to read specific text from the scene.
SCENE_TEXT_RE = re.compile(
    r"^\s*(what\s+(number|time|brand|name|letter|word)|how much|how many)\b",
    re.IGNORECASE,
)

# "Lots of text in the image" cutoff.  Empirically the ~top quartile of
# ``ocr_token_count`` on the TextVQA train pool (see slice_experiment).
TEXT_DENSE_OCR_THRESHOLD = 16

SLICES = ("scene_text", "text_dense", "random")


def matches(slice_name: str, question: str, ocr_tokens) -> bool:
    """True if a row belongs to ``slice_name``."""
    if slice_name == "random":
        return True
    if slice_name == "scene_text":
        return bool(SCENE_TEXT_RE.search(question or ""))
    if slice_name == "text_dense":
        return len(ocr_tokens or []) >= TEXT_DENSE_OCR_THRESHOLD
    raise ValueError(f"unknown slice {slice_name!r}; expected one of {SLICES}")
