"""Evaluate intent routing on a tiny synthetic dataset."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from app.domain.intents import IntentLabel
from app.llm.mock import MockLLM
from app.services.intent_router import IntentRouter


def main() -> dict:
    cases = json.loads(Path("evaluation/datasets/intent_cases.json").read_text(encoding="utf-8"))
    router = IntentRouter(MockLLM())
    labels = [label.value for label in IntentLabel]
    confusion = {gold: Counter() for gold in labels}
    correct = 0
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for case in cases:
        pred = router.route(case["text"]).label.value
        gold = case["label"]
        confusion[gold][pred] += 1
        correct += int(pred == gold)
        for label in labels:
            per_label[label]["tp"] += int(pred == label and gold == label)
            per_label[label]["fp"] += int(pred == label and gold != label)
            per_label[label]["fn"] += int(pred != label and gold == label)
    recalls = {}
    f1s = []
    for label in labels:
        tp, fp, fn = per_label[label]["tp"], per_label[label]["fp"], per_label[label]["fn"]
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0
        recalls[label] = recall
        f1s.append(f1)
    report = {
        "dataset": "tiny synthetic intent set",
        "count": len(cases),
        "accuracy": correct / len(cases),
        "macro_f1": sum(f1s) / len(f1s),
        "per_class_recall": recalls,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items() if v},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
