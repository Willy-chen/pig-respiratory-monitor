# Pig Respiratory Monitor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20084289.svg)](https://doi.org/10.5281/zenodo.20084289)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code, pre-computed AST embeddings, and evaluation results for the paper:

> **Hybrid AST-XGBoost Architecture for Pig Respiratory Health Monitoring in Noisy Farm Environments**
> Wei-Yu Chen, Chao-Wei Huang, Yu-Cheng Xu, Jyh-Shing Roger Jang.
> Submitted to *Computers and Electronics in Agriculture*, 2026.

## Headline result

Under a strictly nested Pig-Level Leave-One-Out cross-validation protocol, the proposed AST-XGBoost classifier achieves a Macro F1 of **0.8713** (95 % cluster-bootstrap CI by pigs: **[0.7740, 0.9086]**) on a 16-pig held-out set. It significantly outperforms a self-supervised BEATs baseline (paired-permutation *p* = 0.017) and is numerically above a fine-tuned AlexNet baseline within overlapping confidence intervals.

## What is in this repository

```
.
├── data/
│   └── features/
│       └── features_3layer_mean.pkl   # cached 768-d AST embeddings + labels + pig IDs
├── src/
│   ├── ast_finetune/                  # AST fine-tuning on pig respiratory audio
│   ├── pipeline/                      # AST + XGBoost grid-search pipeline (leaky-grid version)
│   ├── nested_cv/                     # nested LOOCV protocol used for the paper headline
│   ├── baselines/                     # 9 contemporary baselines re-implemented for parity
│   ├── classifiers/                   # 12-classifier downstream-head benchmark
│   └── stress_tests/                  # SNR robustness + few-shot scaling experiments
├── results/
│   ├── nested_cv/                     # per-segment predictions, fold choices, summary
│   ├── analysis/                      # bootstrap CIs, paired-permutation tests, boxplots
│   ├── baseline_probs/                # per-segment soft-max probabilities for AlexNet & BEATs
│   ├── stress_tests/                  # SNR + few-shot CSVs
│   └── classifier_benchmark.csv       # 12-classifier downstream-head benchmark results
├── figures/                           # publication-quality figures used in the paper
├── docs/                              # architecture notes + reproduction instructions
├── requirements.txt
├── reproduce.sh                       # one-shot reproduction from cached embeddings
├── LICENSE                            # MIT
└── README.md                          # this file
```

## Quick reproduction (~5 min, CPU only)

You don't need to retrain the AST or the baselines to reproduce the paper headline — the AST-XGB pipeline runs from cached AST embeddings and is fully deterministic given the same XGBoost seed.

```bash
git clone https://github.com/Willy-chen/pig-respiratory-monitor.git
cd pig-respiratory-monitor
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` does the following on a single CPU:
1. Loads `data/features/features_3layer_mean.pkl` (1553 segments × 768-d embeddings + labels + pig IDs).
2. Runs the nested LOOCV protocol described in §2.6 of the paper (16 outer × 9 weight cells × 15 inner LOO + 12 threshold cells per cell).
3. Computes the cluster-bootstrap 95 % CI by pigs (10 000 iterations).
4. Computes the paired-permutation tests against the cached AlexNet and BEATs predictions in `results/baseline_probs/`.
5. Writes the per-pig boxplot to `results/analysis/per_pig_f1_boxplot.png` and a fresh `results/analysis/summary.json`.

Expected runtime: ≈ 60 minutes (mostly the inner XGBoost LOOCV; CPU-bound).

## Full reproduction including AST fine-tuning (~6 GPU-hours)

To reproduce from raw audio you also need:
1. The raw stethoscope audio dataset, which is not publicly available due to confidentiality with the participating commercial pig farm. Access may be requested from Prof. Chao-Wei Huang (`cwhuang@mail.npust.edu.tw`) at the NPUST Animal Nutrigenomics Laboratory under a data-use agreement.
2. The fine-tuned AST checkpoint, available at <https://huggingface.co/willychenwii/pig-condition-ast-finetuned>.
3. The official BEATs `BEATs_iter3_plus_AS2M.pt` checkpoint for the BEATs baseline (download per the upstream BEATs repository).
4. The ESC-50 dataset for the SNR stress-test out-of-distribution noise tier (see <https://github.com/karolpiczak/ESC-50>).

With those in place, see `docs/REPRODUCE.md` for end-to-end instructions.

## Citation

If you use this code, please cite our paper (BibTeX entry will be added on acceptance):

```bibtex
@article{chen2026pig,
    author  = {Chen, Wei-Yu and Huang, Chao-Wei and Xu, Yu-Cheng and Jang, Jyh-Shing Roger},
    title   = {Hybrid AST-XGBoost Architecture for Pig Respiratory Health Monitoring in Noisy Farm Environments},
    journal = {Computers and Electronics in Agriculture},
    year    = {2026},
    note    = {under review},
    note    = {Code and dataset: \url{https://doi.org/10.5281/zenodo.20084289}}
}
```

## License

MIT — see `LICENSE`.

## Acknowledgements

This work was supported by the National Science and Technology Council, Taiwan (NSTC 113-2640-B-002-001-) under the integrated AI-and-Smart-Agriculture platform project (PI: Yen-Wen Lu, NTU Biomechatronic Engineering).

We thank the staff of the participating commercial pig farm for facilitating data collection, and the two blinded veterinary acoustic raters and the senior adjudicator (names withheld on request) who performed the secondary expert re-evaluation of model outputs.

The Audio Spectrogram Transformer is from Gong, Chung, and Glass (2021). The BEATs model is from Chen *et al.* (2023). XGBoost is from Chen and Guestrin (2016). The pre-trained Whisper-Tiny encoder is from OpenAI. We acknowledge the maintainers of the `noisereduce`, `transformers`, and `librosa` open-source libraries.
