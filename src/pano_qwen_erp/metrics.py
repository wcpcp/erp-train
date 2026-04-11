from __future__ import annotations

import re
from typing import Dict, Optional

from transformers import EvalPrediction

from swift.metrics.base import EvalMetrics
from swift.utils import Serializer

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_OPTION_HINT_RE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-Za-z0-9]{1,4})\)?",
    re.IGNORECASE,
)
_SINGLE_TOKEN_RE = re.compile(r"^\s*\(?([A-Za-z0-9]{1,4})\)?[\.\):\]\}>]*\s*$")
_LETTER_TOKEN_RE = re.compile(r"\b([A-Z])\b")
_DIGIT_TOKEN_RE = re.compile(r"\b(\d{1,4})\b")


def _normalize_text(text: str) -> str:
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_key(text: str) -> str:
    text = _normalize_text(text).strip("`*_~\"' \t\r\n")
    return text.upper()


def _extract_option_key(text: str) -> Optional[str]:
    if not text:
        return None

    normalized = _normalize_text(text)
    if not normalized:
        return None

    for pattern in (_OPTION_HINT_RE, _SINGLE_TOKEN_RE):
        match = pattern.search(normalized)
        if match:
            return match.group(1).upper()

    for pattern in (_LETTER_TOKEN_RE, _DIGIT_TOKEN_RE):
        match = pattern.search(normalized.upper())
        if match:
            return match.group(1).upper()

    first_token = normalized.split(" ", 1)[0].strip("`*_~()[]{}<>:;,.!?")
    if first_token and len(first_token) <= 4 and first_token.isalnum():
        return first_token.upper()
    return None


class PanoramaMCQMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids
        total = int(preds.shape[0])
        correct = 0
        parsed = 0
        exact = 0

        for i in range(total):
            pred_text = Serializer.from_tensor(preds[i])
            label_text = Serializer.from_tensor(labels[i])

            pred_key = _extract_option_key(pred_text)
            label_key = _normalize_key(label_text)

            if pred_key is not None:
                parsed += 1
            if _normalize_key(pred_text) == label_key:
                exact += 1
            if pred_key == label_key:
                correct += 1

        if total == 0:
            return {"mc_acc": 0.0, "parse_rate": 0.0, "exact_match": 0.0}

        return {
            "mc_acc": round(correct / total, 6),
            "parse_rate": round(parsed / total, 6),
            "exact_match": round(exact / total, 6),
        }
