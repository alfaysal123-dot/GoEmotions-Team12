# Team 12 – Do typos fool emotion classifiers? (GoEmotions)

CSE 484 / 654 NLP, Gebze Technical University. Team: Lara Helva, Fatma Özgül Alkan, Rümeysa Kalay, Tarık Saeede.

**Question.** When about 10% of eligible words in the GoEmotions test set get a synthetic typo, does a fine-tuned DistilBERT lose less macro-F1 than a TF-IDF + logistic regression baseline? Answer so far: **no** (advantage +0.01 ± 0.07 points over 3 training seeds; every 95% bootstrap interval includes zero).

## Reproduce

Baseline row (one command, CPU, under 1 minute):

```
pip install -r requirements.txt && python baseline.py
```

Expected: clean macro-F1 0.4473, typo macro-F1 0.4320, drop 1.53 points.

Everything else is in one Colab notebook, `GoEmotions_Team12.ipynb` (T4 GPU):

| Section | Content |
|---|---|
| ADIM 0–4.5 | Data, typo generator, baseline, 3 typo draws |
| ADIM 4.6 | Typo-generator tests, leakage and label-distribution checks |
| ADIM 4.7 | Baseline error analysis |
| ADIM 4.8 | Spell-check pilot |
| ADIM 5–9 | Preliminary DistilBERT run (no fixed training seed; not in the results table) |
| ADIM 5.3 | DistilBERT, 3 training seeds + typo-rate ablation (5 / 10 / 20%) – results table |
