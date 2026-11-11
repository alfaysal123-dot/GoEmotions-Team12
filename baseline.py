"""Team 12 – GoEmotions typo robustness: TF-IDF + logistic regression baseline.

One command reproduces the baseline row of the results table:

    python baseline.py

It downloads the official GoEmotions split (43,410 / 5,426 / 5,427 comments),
trains TF-IDF + one-vs-rest logistic regression on clean training text, picks one
decision threshold on the clean dev set, and evaluates on the frozen clean test set
and on the same test set with ~10% synthetic typos (3 typo seeds).
Outputs go to results/.
"""
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DATA_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
DATA_DIR = Path("data")
OUT_DIR = Path("results")
TYPO_RATE = 0.10
NOISE_SEEDS = [0, 1, 2]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def load_split(name):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.tsv"
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL + f"{name}.tsv", path)
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels", "id"],
                     quoting=3, keep_default_na=False, dtype=str)
    df["labels"] = df["labels"].apply(lambda s: [int(x) for x in s.split(",")])
    return df


def label_names():
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "emotions.txt"
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL + "emotions.txt", path)
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def add_typos(text, rate=TYPO_RATE, seed=42):
    """Same generator as the notebook (ADIM 3): each word with 4+ letters gets one
    typo (delete, repeat or swap a letter) with probability `rate`. The first letter
    and placeholders such as [NAME] are never changed; a swap at the end can move
    the last letter (jump -> jupm)."""
    rng = random.Random(seed)
    parts = re.findall(r"\[[^\[\]\n]*\]|[A-Za-z]+|[^A-Za-z\[\]]+|[\[\]]", text)
    changed = eligible = 0
    for j, w in enumerate(parts):
        if w.startswith("[") and w.endswith("]"):
            continue
        if not re.fullmatch(r"[A-Za-z]{4,}", w):
            continue
        eligible += 1
        if rng.random() >= rate:
            continue
        i = rng.randint(1, len(w) - 2)
        kind = rng.choice(["sil", "tekrar", "yer_degistir"])  # delete / repeat / swap
        if kind == "sil":
            new = w[:i] + w[i + 1:]
        elif kind == "tekrar":
            new = w[:i] + w[i] + w[i:]
        else:
            new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if new != w:
            parts[j] = new
            changed += 1
    return "".join(parts), changed, eligible


def typo_seed(g, i):
    # seed 0 uses the comment index; seeds 1 and 2 use g*1_000_003 + index (as in the notebook)
    return i if g == 0 else g * 1_000_003 + i


def macro(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def main():
    t0 = time.time()
    names = label_names()
    train, dev, test = load_split("train"), load_split("dev"), load_split("test")
    print(f"train / dev / test: {len(train):,} / {len(dev):,} / {len(test):,}")
    mlb = MultiLabelBinarizer(classes=list(range(len(names))))
    y_train = mlb.fit_transform(train["labels"])
    y_dev = mlb.transform(dev["labels"])
    y_test = mlb.transform(test["labels"])

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(train["text"])
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced",
                                                 random_state=42), n_jobs=1)
    clf.fit(X_train, y_train)

    # threshold chosen on the clean dev set only (grid 0.10-0.85, step 0.05)
    p_dev = clf.predict_proba(tfidf.transform(dev["text"]))
    grid = np.round(np.arange(0.10, 0.851, 0.05), 2)
    dev_scores = [(float(t), macro(y_dev, (p_dev >= t).astype(int))) for t in grid]
    thr, dev_f1 = max(dev_scores, key=lambda x: x[1])
    print(f"dev: threshold {thr:.2f}, macro-F1 {dev_f1:.4f}")

    texts = list(test["text"])
    p_clean = clf.predict_proba(tfidf.transform(texts))
    pred_clean = (p_clean >= thr).astype(int)
    clean_f1 = macro(y_test, pred_clean)

    rows, typo_texts, typo_preds = [], {}, {}
    for g in NOISE_SEEDS:
        out = [add_typos(t, seed=typo_seed(g, i)) for i, t in enumerate(texts)]
        typo_texts[g] = [o[0] for o in out]
        typo_preds[g] = (clf.predict_proba(tfidf.transform(typo_texts[g])) >= thr).astype(int)
        f1 = macro(y_test, typo_preds[g])
        rows.append({"typo_seed": g, "changed_words": sum(o[1] for o in out),
                     "eligible_words": sum(o[2] for o in out),
                     "changed_comments": sum(o[1] > 0 for o in out),
                     "typo_macro_f1": f1, "drop_points": 100 * (clean_f1 - f1)})
    seeds = pd.DataFrame(rows)

    # 95% bootstrap interval for clean test macro-F1 and for the seed-0 drop
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(texts)
    boot_clean, boot_drop = [], []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        c = macro(y_test[idx], pred_clean[idx])
        boot_clean.append(c)
        boot_drop.append(c - macro(y_test[idx], typo_preds[0][idx]))
    ci_clean = np.percentile(boot_clean, [2.5, 97.5])
    ci_drop = np.percentile(boot_drop, [2.5, 97.5]) * 100

    OUT_DIR.mkdir(exist_ok=True)
    seeds.to_csv(OUT_DIR / "baseline_typo_seeds.csv", index=False)
    pd.DataFrame({"text": texts, "text_typo": typo_texts[0],
                  "gold": [";".join(names[j] for j in np.where(r)[0]) for r in y_test],
                  "pred_clean": [";".join(names[j] for j in np.where(r)[0]) for r in pred_clean],
                  "pred_typo": [";".join(names[j] for j in np.where(r)[0]) for r in typo_preds[0]]}
                 ).to_csv(OUT_DIR / "baseline_test_predictions.csv", index_label="test_index")
    summary = {"dev_threshold": thr, "dev_macro_f1": dev_f1, "test_clean_macro_f1": clean_f1,
               "test_clean_macro_f1_95ci": ci_clean.tolist(),
               "test_typo_macro_f1_seed0": float(seeds.typo_macro_f1[0]),
               "drop_points_seed0": float(seeds.drop_points[0]),
               "drop_points_seed0_95ci": ci_drop.tolist(),
               "drop_points_mean_3_typo_seeds": float(seeds.drop_points.mean()),
               "drop_points_std_3_typo_seeds": float(seeds.drop_points.std(ddof=1)),
               "runtime_s": round(time.time() - t0, 1)}
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print(seeds.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
