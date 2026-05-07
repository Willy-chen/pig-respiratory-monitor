# Baseline Model Specifications

## Data & Evaluation Protocol

> **Same data and protocol as `20260302_ultimate`.**

| Setting | Value |
|---|---|
| Dataset | `get_full_dataset()` from `20260302_ultimate/data_utils.py` |
| Evaluation set | **XGB_SET** — 50% stratified file-level holdout (same seed, same split as ultimate method) |
| Protocol | **File-level Leave-One-Out Cross Validation (LOOCV)** |
| Audio SR | 16 kHz |
| Segment length | 10 seconds (padded/truncated) |
| Classes | 0 = No-Breathing, 1 = Normal, 2 = Abnormal |
| Primary metric | Macro F1-score |

The `20260302_ultimate` method uses **AST features** (Audio Spectrogram Transformer, last 3 hidden layers, mean pooled) extracted from a fine-tuned AST model, then fed to **XGBoost with LOOCV** and grid-searched class weights + decision thresholds. It achieves a Macro F1 of **0.8894** on the standardized benchmark. All baselines are evaluated on the identical XGB_SET under the identical LOOCV protocol for a fair comparison.

---

## Baseline 1 — Yin et al. (2021)

**Paper:** *Recognition of sick pig cough sounds based on convolutional neural network in field situations*, Information Processing in Agriculture, Vol. 8(3), pp. 369–379.

**Paper Quote:**
> "The proposed method converts audio signals into spectrogram images and then applies AlexNet for feature learning and classification."

### Model: `SpectrogramAlexNet`

| Component | Detail |
|---|---|
| Feature | On-the-fly `torchaudio.MelSpectrogram`: `n_mels=227`, `n_fft=1024`, `hop_length=512` |
| Pre-processing | Log₁₀ scaling, per-image instance normalization |
| Reshape | `AdaptiveAvgPool2d → (227, 227)`, repeated 3× for RGB-compatibility |
| Backbone | **AlexNet** (ImageNet pre-trained, `torchvision.models.AlexNet_Weights.DEFAULT`) |
| Classifier | Default AlexNet classifier with final `Linear(4096 → 3)` |

**Architecture Summary:**
```
Conv(11×11, stride=4, 96) → MaxPool → LRN
Conv(5×5, 256) → MaxPool → LRN
Conv(3×3, 384) → Conv(3×3, 384) → Conv(3×3, 256) → MaxPool
FC(4096) → FC(4096) → FC(num_classes)
```

**Macro F1-Score: 0.8811** (Updated: +0.1043 improvement with AST set)

---

## Baseline 2 — Shen et al. (2022)

**Paper:** *Fusion of acoustic and deep features for pig cough sound recognition*, Computers and Electronics in Agriculture, Vol. 197, p. 106994.

**Paper Quote:**
> "We propose a fusion framework that combines traditional acoustic features (CQT, STFT) with deep features extracted by a LeNet-5 inspired CNN, and feeds the fused embeddings into an SVM for final classification."

### Model: `LeNet5Fusion` → SVM

| Component | Detail |
|---|---|
| Feature | `torchaudio.Spectrogram`: `n_fft=1024`, `hop_length=512` (STFT proxy) |
| Phase 1 | LeNet-5 CNN pre-trained with Cross-Entropy to learn embeddings |
| Phase 2 | Embeddings extracted → Multi-class SVM (`RBF kernel`, `C=1.0`, `gamma='scale'`) |

**Architecture Summary:**
```
Input (1×freq×time)
Conv(5×5, 6) → ReLU → MaxPool(2×2)
Conv(5×5, 16) → ReLU → AdaptiveAvgPool → (16×10×10)
FC(1600 → 120) → ReLU → FC(120 → 84) → ReLU → FC(84 → 3)
```

**Macro F1-Score: 0.7080** (Updated: +0.1253 improvement)

---

## Baseline 3 — Wu et al. (2022)

**Paper:** *Combined spectral and speech features for pig speech recognition*, PLOS ONE, Vol. 17(12), e0276778.

**Paper Quote:**
> "We apply RNN, LSTM, and GRU networks to process multi-level mel-cepstral (MLMC) feature sequences extracted from pig vocalizations."

### Model: `MFCC_LSTM`

| Component | Detail |
|---|---|
| Feature | `torchaudio.MFCC`: `n_mfcc=40`, `n_mels=128`, `sample_rate=16000` |
| Input shape | `(batch, time_steps, 40)` |
| Architecture | 2-layer Bidirectional LSTM, `hidden_size=64` |

**Architecture Summary:**
```
MFCC (40 coeffs/frame) → Bi-LSTM (2 layers, hidden=64)
→ Concat [h_forward, h_backward] → FC(128→64) → ReLU → FC(64→3)
```

**Macro F1-Score: 0.6451** (Updated: +0.0400 improvement)

---

## Baseline 4 — Hou et al. (2024)

**Paper:** *Study on a pig vocalization classification method based on multi-feature fusion*, Sensors, Vol. 24(2), p. 313.

**Paper Quote:**
> "Multiple acoustic features extracted from different time and frequency domains are fused and fed into a BP (backpropagation) neural network for pig vocalization classification."

### Model: `MultiFeatureMLP`

| Component | Detail |
|---|---|
| Features (Time domain) | MFCC: `n_mfcc=40` — mean, variance, max across frames = 120 values |
| Features (Freq domain) | Spectrogram: `n_fft=1024` — mean, variance, max across frames = 1539 values |
| Total input dim | **1659** (`120 + 1539`) |
| Architecture | 3-layer BPNN (512-128-3) with BatchNorm and Dropout |

**Architecture Summary:**
```
Input (1659)
→ FC(1659→512) → BatchNorm → ReLU → Dropout(0.3)
→ FC(512→128) → BatchNorm → ReLU
→ FC(128→3)
```

**Macro F1-Score: 0.3466** (Updated: +0.0653 improvement)

---

## Baseline 5 — Sheikh et al. (2024)

**Paper:** *Bird Whisperer: Leveraging large pre-trained acoustic model for bird call classification*, Interspeech 2024, pp. 5028–5032.

**Paper Quote:**
> "We leverage the Whisper encoder as a frozen backbone and attach a lightweight CNN followed by a fully connected network (FCN) to classify the target audio categories."

### Model: `WhisperClassifier`

| Component | Detail |
|---|---|
| Feature | Raw waveform → Whisper Log-Mel Spectrogram (80 bins) |
| Backbone | OpenAI **Whisper Tiny** Encoder (frozen) |
| Classifier | Conv1d(384→128) → MaxPool → FC(128→64→3) |

**Architecture Summary:**
```
Raw Audio → Whisper Tiny Encoder (frozen)
→ hidden (batch, seq, 384)
→ Conv1d(384→128, k=3) → ReLU → AdaptiveAvgPool → (batch, 128)
→ FC(128→64) → ReLU → FC(64→3)
```

**Macro F1-Score: 0.6851** (Updated: +0.0458 improvement)

---

## Baseline 6 — Dörr et al. (2026)

**Paper:** *Alarming pig vocalization-based prediction using the self-supervised BEATs model*, 2026.

**Paper Quote:**
> "We use the BEATs (Bidirectional Encoder representation from Audio Transformers) self-supervised model pre-trained on large unlabeled corpora. The model is then fine-tuned on pig farm audio."

### Model: `OfficialBEATs`

| Component | Detail |
|---|---|
| Architecture | **BEATs Iter3** (Iterative self-supervised Audio Transformer) |
| Checkpoint | `BEATS_iter3_plus_AS2M.pt` (Official Microsoft weights) |
| Backbone | 12 transformer layers, 768-dim embeddings, 12 attention heads |
| Classifier | Linear head (768 → 3) on the mean-pooled global embedding |

**Macro F1-Score: 0.8567** (Updated: +0.0732 improvement)

---

## Baseline 7 — Wang et al. (2026)

**Paper:** *A CNN-SVM Study Based on the Fusion of Spectrogram and Thermal Imaging for Pig Cough Recognition (CoughRNet)*, 2026.

**Paper Quote:**
> "CoughRNet uses early fusion to combine deep acoustic features from spectrograms and visual features from thermal imaging, which are extracted by parallel CNN branches and fused before SVM classification."

### Model: `CoughRNet` → SVM

| Component | Detail |
|---|---|
| Feature | MelSpectrogram (dual-branch proxy for multimodal fusion) |
| CNN Branch | 2-layer CNN (16-32 filters) extracting 512-dim features per branch |
| Final phase | Flattened fusion (1024-dim) → Multi-class SVM (RBF) |

**Architecture Summary:**
```
Spectrogram → CNN_acoustic ┐
Spectrogram → CNN_visual   ┘ → Concat(1024) → SVM (RBF)
```

**Macro F1-Score: 0.5906** (Updated: +0.0437 improvement)

---

## Baseline 8 — MDPI/ResearchGate (2026)

**Paper:** *Swine Disease Classification Using Deep Learning (PigCough-CNN)*, MDPI, 2026.

**Paper Quote:**
> "PigCough-CNN uses 8 MFCCs and spectrogram features as dual inputs, processed by parallel CNN streams whose features are fused before classification."

### Model: `PigCoughCNN`

| Component | Detail |
|---|---|
| Feature 1 | STFT Spectrogram (`n_fft=1024`) |
| Feature 2 | Low-order MFCCs (`n_mfcc=8`) |
| Fusion | Parallel CNN encoders (32 filters each) → FC(128) → FC(3) |

**Architecture Summary:**
```
Spectrogram → CNN_spec → flatten (512)  ┐
MFCC (8 coef) → CNN_mfcc → flatten (512) ┘ → Concat(1024) → FC→ReLU→DropOut → FC(3)
```

**Macro F1-Score: 0.6961** (Updated: +0.0472 improvement)

---

## Baseline 9 — Nithinkumar et al. (2026)

**Paper:** *Investigation into respiratory sound classification for an imbalanced data set using hybrid LSTM-KAN architectures*, 2026.

**Paper Quote:**
> "We propose a hybrid LSTM-KAN model that combines LSTM's temporal sequence modelling with Kolmogorov-Arnold Networks (KAN) to better handle non-linear, imbalanced datasets."

### Model: `LSTMKAN`

| Component | Detail |
|---|---|
| Feature | MFCC (40) sequence |
| RNN | 2-layer LSTM (hidden=64) |
| Classifier | **KAN (Kolmogorov-Arnold Network)**: `[64 → 32 → 3]` |

**Architecture Summary:**
```
MFCC (40 frames) → LSTM (2 layers, h=64)
→ last hidden state (64)
→ KAN [64 → 32 → 3]
```

**Macro F1-Score: 0.6443** (Updated: +0.1413 improvement with official KAN)

---

## Results Summary Table

| Rank | Method | Feature | Model | Macro F1 |
|---|---|---|---|---|
| 0 | **20260302_ultimate** | AST (3-layer pool) | **XGBoost** | **0.8894** |
| 1 | Yin 2021 | MelSpec | AlexNet | 0.8811 |
| 2 | Dörr 2026 | Audio | **BEATs Iter3** | 0.8567 |
| 3 | Shen 2022 | STFT | LeNet-5 + SVM | 0.7080 |
| 4 | MDPI 2026 | Spec + MFCC(8) | Dual-CNN | 0.6961 |
| 5 | Sheikh 2024 | Log-Mel | Whisper-Tiny + CNN | 0.6851 |
| 6 | Wu 2022 | MFCC(40) | Bi-LSTM | 0.6451 |
| 7 | Nithin 2026 | MFCC(40) | LSTM-KAN | 0.6443 |
| 8 | Wang 2026 | MelSpec | CoughRNet (CNN-SVM) | 0.5906 |
| 9 | Hou 2024 | Stats (1659) | BP-MLP | 0.3466 |
