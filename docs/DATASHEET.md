# Datasheet for the Pig Respiratory AST-Embedding Dataset

Following Gebru *et al.* (2021) *Datasheets for Datasets*. We document only the publicly-released **AST-embedding subset** here; the underlying raw audio is private and is not described in detail.

---

## Motivation

**For what purpose was the dataset created?**
To enable reproduction of the AST-XGBoost respiratory-classification results in Chen *et al.* (2026), without releasing the underlying audio (which is restricted by farm-confidentiality agreements).

**Who created the dataset and on behalf of which entity?**
Wei-Yu Chen (NTU CSIE) and Chao-Wei Huang (NPUST Animal Nutrigenomics Lab) collected the underlying audio. The AST embeddings were extracted by Wei-Yu Chen using the fine-tuned AST released as `willychenwii/pig-condition-ast-finetuned` on Hugging Face.

**Who funded the creation of the dataset?**
National Science and Technology Council, Taiwan (NSTC 113-2640-B-002-001-), under the integrated AI-and-Smart-Agriculture platform project (PI: Yen-Wen Lu, NTU Biomechatronic Engineering).

---

## Composition

**What do the instances represent?**
Each instance is a 768-dimensional embedding vector representing one 10-second segment of audio from a single pig. Embeddings come from the **mean of the last three transformer layers** (layers 10, 11, 12) of the fine-tuned Audio Spectrogram Transformer.

**How many instances are there?**
1 553 segments from 16 unique pigs. Class breakdown: No-Breathing 533 (34.3 %), Normal 588 (37.9 %), Abnormal 432 (27.8 %).

**Does the dataset contain all possible instances?**
No. Instances are the 16-pig XGB Set described in §3.1.3 of the paper. The complementary 15-pig AST Set (used for AST fine-tuning) is *not* released in embedding form because it would not be useful to downstream users — the AST training labels are file-level only and the embeddings would be a confounded supervised target.

**What data does each instance consist of?**
A 768-dim float32 vector + a 3-class label + a `pig_id` string (the original audio filename, used as the LOOCV grouping key). No personally identifying information about the farm staff or the animals beyond the recording filename.

**Is there a label or target?**
Yes. 3-class label in {0: No-Breathing, 1: Normal, 2: Abnormal}.

**Is any information missing from individual instances?**
Yes. The original timestamps (`start_seconds`, `end_seconds`) within each source audio file are removed because they could in principle help re-identify the recording sessions. Within-pig segment ordering is preserved by `segment_idx`.

**Are relationships between individual instances made explicit?**
Yes — segments belonging to the same pig share a `pig_id`. This is the *only* grouping signal exposed and is required for proper Pig-Level LOOCV.

**Are there recommended data splits?**
Yes — 16-fold Pig-Level Leave-One-Out cross-validation. Fold definitions are in `data/loocv_folds.csv` and `data/splits/fold_NN/`.

**Are there errors, sources of noise, or redundancies in the dataset?**
- **Label noise**: Section 4.3 of the paper reports a secondary blinded expert re-rating which suggested that approximately 39.8 % of the model's flagged "errors" in the No-Breathing class were actually biological sounds (faint normal breathing, grunts, pathological exhalations) under-labelled by the original annotators. This applies to the underlying audio, not the embeddings, but propagates to the labels released here. We recommend treating Macro F1 as a lower bound on real-world precision.
- **Class imbalance**: the Abnormal class is drawn from approximately 13 source recordings; some pigs in the LOOCV split contribute zero Abnormal segments (see `data/loocv_folds.csv`).
- **Single-farm bias**: all recordings are from one commercial pig farm. Generalisation across barn ventilation designs, breed compositions, and seasonal variation is not validated.

**Does the dataset rely on external resources?**
The embeddings depend on the fine-tuned AST checkpoint (`willychenwii/pig-condition-ast-finetuned` on Hugging Face). They cannot be regenerated without that checkpoint or without the original raw audio.

**Does the dataset contain confidential or sensitive data?**
The released embeddings do **not** contain personally identifying information or any direct audio samples. The original audio is confidential per a farm agreement and is not redistributed.

---

## Collection

**How was the underlying audio collected?**
Electronic stethoscopes were placed non-invasively on the backs of pigs in a commercial barn. Each pig was recorded in a single session. No procedure was performed beyond passive acoustic observation.

**Over what timeframe was the data collected?**
Two collection sessions: one in 2024 and one in 2025 (pig IDs starting with `2024…` and `2025…` correspond to the second session; pig IDs starting with `114-…` correspond to the first session).

**Were any ethical review processes conducted?**
The animal-use protocol is being verified with the relevant institutional animal-care committee for retroactive exemption review prior to publication; see the Animal Ethics Statement in the paper.

---

## Preprocessing / Cleaning / Labelling

**Was any preprocessing of the raw audio done?**
1. Bandpass filter 50–3 000 Hz (preserves respiratory fundamentals + first formants).
2. Spectral-gating denoise via the `noisereduce` Python library, gate threshold −20 dB.
3. 10-second segmentation @ 16 kHz mono. For Normal/Abnormal classes, segmentation hops 5 s through expert-annotated breathing intervals; for No-Breathing it hops 30 s through the gap intervals.

**How were labels assigned?**
Two expert raters annotated start/end timestamps for each breathing event in the foundational 12 recordings; events labelled "Normal" or "Abnormal". Segments outside any breathing event are class No-Breathing.

In addition, 11 weakly-labelled "predominantly Normal" recordings (file-level label only) and 8 strongly-labelled Abnormal recordings (precise timestamps) were added to balance the class distribution.

A secondary blinded re-rating of model-flagged disagreements was performed by two veterinary acoustic raters with adjudication by a third senior rater (n = 191 segments); see §4.3 of the paper.

---

## Uses

**Has the dataset been used for any tasks already?**
Yes — for the experiments reported in Chen *et al.* (2026), specifically AST fine-tuning, AST + XGBoost classification, 9-baseline architectural comparison, 12-classifier downstream-head benchmark, SNR robustness stress test, and few-shot data-fraction scaling.

**What other tasks could this dataset be used for?**
- Drop-in benchmarking of new linear / tree-based / shallow-MLP classification heads on stethoscope audio embeddings.
- Studying the small-sample per-pig variance regime (≤ 16 unique subjects).
- Probing AST embedding geometry for medical / animal-bioacoustic transfer.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed that might affect future uses?**
- Single farm = limited generalisation evidence.
- The 16-pig LOOCV evaluation has high per-fold variance; do not interpret single-seed comparisons as reliable. Use cluster-bootstrap CIs.
- Embeddings come from a model fine-tuned on the AST set; they are not "neutral" pre-trained features. A user wanting unbiased pretrained embeddings should re-extract from the public `MIT/ast-finetuned-audioset-10-10-0.4593` instead of using these embeddings.

**Are there tasks for which the dataset should not be used?**
Clinical decision-making in production veterinary settings without re-validation on the target farm. The dataset is intended as a research benchmark, not a deployment-ready surveillance system.

---

## Distribution

**Will the dataset be distributed to third parties?**
Yes — the AST-embedding subset is released under MIT license alongside the code in this repository. The underlying raw audio is *not* redistributed; access can be requested under a data-use agreement.

**How will the dataset be distributed?**
GitHub repository: <https://github.com/Willy-chen/pig-respiratory-monitor>. Permanent archival DOI: <https://doi.org/10.5281/zenodo.20084290>.

**When will the dataset be distributed?**
On acceptance of the paper.

---

## Maintenance

**Who is supporting / hosting / maintaining the dataset?**
Wei-Yu Chen (`d14922032@ntu.edu.tw`); maintenance pull requests welcome. Substantive issues should be raised via GitHub Issues.

**How can the owner / curator be contacted?**
Dataset access: Prof. Chao-Wei Huang (`cwhuang@mail.npust.edu.tw`), NPUST Animal Nutrigenomics Laboratory. Paper corresponding author: Jyh-Shing Roger Jang (`jang@mirlab.org`). First author: Wei-Yu Chen.

**Is there a versioning scheme?**
v1.0 corresponds to the embeddings used in the paper as submitted. Future re-extractions (e.g., with an updated AST checkpoint) will be tagged as v1.1, v2.0, etc.

**If others want to extend / augment / build on / contribute to the dataset, is there a mechanism for them to do so?**
Pull requests and forks are welcome. The MIT license allows redistribution and modification.
