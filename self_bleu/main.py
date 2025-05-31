"""
pip install "sacrebleu[ja]"
"""

from datasets import load_dataset, Dataset
from sacrebleu.metrics import BLEU
import numpy as np

from logging import getLogger, INFO

logger = getLogger(__name__)
logger.setLevel(INFO)


def score(dataset_name):
    ds = load_dataset(dataset_name, split="train")
    bleu = BLEU(trg_lang="ja")
    print(len(ds))
    ds = ds.shuffle(seed=42).select(range(500))

    def calc_score(example):
        target = example["text"]
        sys = [target]
        refs = [[d["text"] for d in ds if d["text"] != target]]

        result = bleu.corpus_score(sys, refs)
        score = result.score
        return {"score": score}

    ds = ds.map(calc_score)

    ## score av
    scores = np.array([d["score"] for d in ds])

    mean = np.mean(scores)
    var = np.var(scores)
    std = np.std(scores)
    max = np.max(scores)
    min = np.min(scores)
    print(
        f"len {len(scores)}, mean: {mean} ,var: {var}, std: {std}, max: {max}, min: {min}"
    )

    th = std
    # th = 100
    for d in ds:
        if d["score"] > th:
            print(d["score"], d["text"])


def main():
    score("if001/elementray_l")


if __name__ == "__main__":
    main()
