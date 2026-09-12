# Data-Efficient Multilingual Vision-Language Modeling for Low-Resource Languages

This project is an empirical study of parameter-efficient (QLoRA) adaptation of a pretrained
multilingual vision-language model to a low-resource language, using Tigrinya
as a case study.

## Research question

How effectively can a pretrained multilingual vision-language model be
adapted to Tigrinya with limited multimodal supervision, and how do
multimodal input and training-data size affect generation quality?

## Model

- Google Gemma 3 4B Instruct
- 4-bit NF4 quantization (bitsandbytes)
- LoRA / QLoRA parameter-efficient fine-tuning (r=16, alpha=32, dropout=0.05)

## Dataset

This project uses the Tigrinya (`tir`) configuration of the
[LaMuN](https://huggingface.co/datasets/tharindu/LaMuN) multilingual
image-caption dataset — 3,028 examples from 1,768 unique articles. Because
multiple images can come from the same article, splitting is done at the
**article level**, not the example level, to prevent leakage across
train/validation/test. An explicit overlap check in the notebook confirms
zero shared articles across all three splits. See the LaMuN dataset page for
licensing and source-content terms.

## Experiments

1. **Zero-shot baseline** — text-only / image-only / image+title prompting, no fine-tuning
2. **Multimodal QLoRA** — fine-tuned on image + title → caption
3. **Text-only QLoRA ablation** — fine-tuned on title → caption only (no image), isolating the image's contribution
4. **Training-data-size ablation** — multimodal QLoRA retrained from scratch on 100 / 500 / 1,000 / 2,440 (full) examples

### Training protocol

An initial full-data exploratory run over 3 epochs showed validation loss
improving through epoch 2 and then increasing slightly in epoch 3 (mild
overfitting): **2.5100 → 2.4232 → 2.4592**. Based on this, experiments 3 and 4
use a fixed **one-epoch** training budget for a controlled, fair comparison
across conditions and data sizes, rather than an arbitrary choice.

## Results

All metrics are computed on the held-out test set (n=288), which is disjoint
from train/validation at the article level.

### Zero-shot baseline (raw metrics)

| Condition     |  BLEU |  chrF | ROUGE-L |
| ------------- | ----: | ----: | ------: |
| Text only     | 0.281 | 6.711 |  0.0028 |
| Image only    | 0.007 | 1.939 |  0.0024 |
| Image + title | 0.095 | 3.883 |  0.0017 |

_(A post-hoc normalization that strips English instruction-following preamble
from generations is also reported in the notebook as a secondary diagnostic;
it does not change the ranking above.)_

### Main comparison: zero-shot vs. text-only vs. multimodal QLoRA

| Experiment               |      BLEU |      chrF |    ROUGE-L |
| ------------------------ | --------: | --------: | ---------: |
| Zero-shot, image + title |     0.095 |     3.883 |     0.0017 |
| Text-only QLoRA          |     1.230 |     7.794 |     0.0023 |
| **Multimodal QLoRA**     | **2.144** | **9.721** | **0.0228** |

### Training-data-size ablation (multimodal QLoRA)

| Training examples | Val. loss |      BLEU |       chrF | ROUGE-L | Tigrinya % | Mean repetition |
| ----------------: | --------: | --------: | ---------: | ------: | ---------: | --------------: |
|               100 |     3.293 |     1.011 |      9.105 |  0.0176 |      78.14 |           0.249 |
|               500 |     2.892 |     1.025 |      6.942 |  0.0226 |      71.12 |           0.146 |
|             1,000 |     2.722 |     1.618 |      9.276 |  0.0104 |      76.50 |           0.102 |
|      2,440 (full) | **2.506** | **1.683** | **10.146** |  0.0156 |      76.49 |           0.123 |

## Findings

1. Parameter-efficient adaptation substantially improves Tigrinya generation
   over zero-shot prompting on all three metrics.
2. Multimodal QLoRA outperforms text-only QLoRA on BLEU, chrF, and ROUGE-L,
   suggesting the image contributes beyond what title-only adaptation
   captures — though this is based on a single run per condition, which
   limits how strong a claim it supports.
3. Validation loss decreases consistently as training data increases from
   100 to 2,440 examples.
4. Test-set lexical metrics (BLEU/chrF/ROUGE-L) vary non-monotonically across
   intermediate data sizes (notably ROUGE-L peaks at 500), but the full
   2,440-example model reaches the strongest BLEU and chrF and the lowest
   mean repetition.
5. Despite improved fluency, qualitative inspection indicates that reliable
   visual grounding and factual consistency remain challenging — the model
   can produce fluent Tigrinya that is not well grounded in the image or
   reference, a gap the automatic metrics above do not capture.

This project is best read as an empirical study of parameter-efficient
adaptation of a multilingual VLM to a low-resource language — not as a claim
of state-of-the-art Tigrinya VLM performance or of reliable visual grounding.

## Limitations

- The Tigrinya subset of LaMuN is small (3,028 examples / 1,768 unique articles), limiting statistical power, especially in the low-data regime (100/500 examples).
- Reference captions are news captions, not pure visual descriptions — they are not always fully inferable from the image alone, which caps achievable scores on lexical-overlap metrics regardless of model quality.
- BLEU/chrF/ROUGE-L measure lexical overlap with a single reference and do not directly measure visual grounding or factual correctness.
- Each data-size condition was trained once (one epoch, one seed) rather than averaged over multiple runs, so the non-monotonic trend across data sizes should be read with that variance in mind.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── notebooks/
│   └── tigrinya_vlm_experiments.ipynb   # full experimental workflow
├── results/
│   ├── zero_shot_baseline_metrics.json
│   ├── multimodal_qlora_metrics.json
│   ├── text_only_qlora_metrics.json
└──   └── data_efficiency_metrics.json
```

## Reproducing

Requires a CUDA GPU (bfloat16 base model + 4-bit NF4 quantized fine-tuning),
and your own Hugging Face token with access to `google/gemma-3-4b-it`. Run
the notebook top-to-bottom in a fresh runtime (**Restart Runtime → Run All**)
rather than relying on cached variables from a prior session.

`OUTPUT_DIR` in the configuration cell controls where predictions/checkpoints are written. It's currently set to `"/content"`, which assumes a Colab runtime — running this notebook elsewhere will require changing that one line to a local/relative folder(e.g. `"results"`) first.
