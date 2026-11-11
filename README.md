# GoEmotions — Team 12

**Do typos fool emotion classifiers?** A CSE 484 / 654 NLP project comparing the robustness of TF-IDF + Logistic Regression and fine-tuned DistilBERT on GoEmotions.

[Open the notebook](GoEmotions_Team12.ipynb)

## Research question

When synthetic character-level spelling errors are introduced with probability 0.10 per eligible word in the GoEmotions test set, does fine-tuned DistilBERT lose less macro-F1 than the TF-IDF + Logistic Regression baseline?

Both models use the same clean training split and the same clean and perturbed test comments. Each model's decision threshold is selected using the clean validation set.

## Dataset and methods

- **Dataset:** GoEmotions, `simplified` configuration; English Reddit comments with 27 emotions plus neutral, supporting multiple labels per comment.
- **Splits:** 43,410 train / 5,426 validation / 5,427 test comments.
- **Baseline:** 20,000 TF-IDF word unigram/bigram features and class-balanced one-vs-rest Logistic Regression.
- **Transformer:** `distilbert-base-uncased`, 28 sigmoid outputs, binary cross-entropy, two fine-tuning epochs.
- **Typos:** deletion, repetition, and adjacent-letter swaps; bracketed placeholders such as `[NAME]` are preserved.
- **Evaluation:** macro-F1 drop, three perturbation draws, three Transformer training seeds, 1,000 paired-bootstrap resamples for each primary seed comparison, and 5% / 10% / 20% typo-rate ablations.

## Recorded results

Rounded values from the notebook's saved outputs, using the primary 10% typo draw:

| Model / training seed | Clean macro-F1 | Typo macro-F1 | 95% CI for robustness advantage, in macro-F1 points |
| --- | ---: | ---: | --- |
| TF-IDF + Logistic Regression | 0.4473 | 0.4320 | — |
| DistilBERT / 0 | 0.4249 | 0.4104 | [-0.78, +0.84] |
| DistilBERT / 1 | 0.4464 | 0.4311 | [-0.95, +0.87] |
| DistilBERT / 2 | 0.4473 | 0.4315 | [-0.97, +0.90] |

Advantage = baseline macro-F1 drop minus DistilBERT macro-F1 drop. All three primary confidence intervals include zero. These runs provide no statistically supported evidence of a smaller DistilBERT drop under the primary comparison.

## Files

| File | Contents |
| --- | --- |
| `GoEmotions_Team12.ipynb` | Research notebook with code, explanations, and saved experiment outputs |
| `requirements.txt` | Dependency inventory inferred from the notebook |

The notebook preserves the uploaded source, including its recorded experiment outputs.

## Reproducing the work

Use Google Colab with a GPU runtime, or a compatible Jupyter environment. The recorded training runs used a Tesla T4.

1. Open `GoEmotions_Team12.ipynb` in the notebook environment.
2. Upload `requirements.txt` and run `%pip install -r requirements.txt` in a setup cell before the first imports.
3. Run ADIM 0–4.5 for the baseline and perturbation draws.
4. Run ADIM 4.6–4.8 for data checks, error analysis, and the spell-check pilot.
5. Run the ADIM 5 preparation cell before ADIM 5.3, which performs the three-seed Transformer experiment and ablations. The earlier single-run Transformer experiment is preliminary.

Dependencies are unpinned; record the working package versions when reproducing the experiment. The notebook contains a `torchvision` troubleshooting cell that should be reviewed for the selected environment.

Keep `GENERATE_DELIVERABLES = False` during experimentation. The notebook's team configuration contains placeholders that must be completed before generating submission documents. Some validation and error-analysis cells have no recorded output in the source snapshot. Report and slide exports still refer to preliminary single-run variables and should be updated before final submission.

Download both `/content/goemotions_outputs` and `/content/GoEmotions_seed_runs` before the Colab runtime ends. The existing ZIP-export cell includes only the first directory.

## Team

Lara Helva · Fatma Özgül Alkan · Rümeysa Kalay · Tarik Saeede

## Data sources

- [GoEmotions on Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [Original GoEmotions repository](https://github.com/google-research/google-research/tree/master/goemotions)
- [Demszky et al. (2020), GoEmotions](https://arxiv.org/abs/2005.00547)

The dataset is identified as Apache-2.0 in the notebook. This repository does not add a license for the team's own code.
