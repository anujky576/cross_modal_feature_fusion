# Feature Fusion — Multi-Modal Binary Classifier

A multi-modal machine learning pipeline that trains three independent models on different input representations (emoticons, text sequences, and deep feature vectors), then fuses them into a single combined classifier.

---

## Project Structure

```
.
├── feature_fusion.py          # Main training & inference script
├── datasets/
│   ├── train/
│   │   ├── train_emoticon.csv
│   │   ├── train_text_seq.csv
│   │   └── train_feature.npz
│   ├── valid/
│   │   ├── valid_emoticon.csv
│   │   ├── valid_text_seq.csv
│   │   └── valid_feature.npz
│   └── test/
│       ├── test_emoticon.csv
│       ├── test_text_seq.csv
│       └── test_feature.npz
├── pred_emoticon.txt          # Output: emoticon model predictions
├── pred_textseq.txt           # Output: text-seq model predictions
├── pred_deepfeat.txt          # Output: deep features model predictions
└── pred_combined.txt          # Output: fused model predictions
```

---

## Pipeline Overview

```
Emoticon CSV  ──► LightGBM         ──►─┐
Text Seq CSV  ──► LSTM (Keras)     ──►─┼──► Feature Fusion ──► LightGBM ──► pred_combined.txt
Feature NPZ   ──► PCA + LightGBM   ──►─┘
```

Each branch produces its own predictions as well as learned feature representations that are concatenated for the final fused model.

---

## Models

### 1. Emoticon Model (`train_emoticon_model`)

**Input:** Unicode emoticon strings (characters in range U+1F600–U+1F721)

**Preprocessing:**
- Counts character frequency across the training set
- Removes the 7 most frequent characters (noise reduction)
- Encodes each remaining character to its Unicode index offset
- Keeps the first 3 encoded characters as features (`encoded_1`, `encoded_2`, `encoded_3`)

**Model:** LightGBM (GBDT)
- Learning rate: `0.45`
- Feature fraction: `0.3`
- Num leaves: `31`
- Rounds: `105`

---

### 2. Text Sequence Model (`train_text_seq_model`)

**Input:** Character-level text strings with a 3-character prefix (e.g. `ID:`, `SQ:`)

**Preprocessing:**
- Strips the first 3 characters from each string
- Character-level tokenization via Keras `Tokenizer`
- Pads/truncates sequences to length `47`

**Model:** LSTM (Keras/TensorFlow)
- Embedding dim: `64`
- LSTM units: `8`
- Dropout: `0.2`
- Output: sigmoid (binary)
- Optimizer: Adam | Loss: Binary Crossentropy
- Epochs: `150`, Batch size: `32`

---

### 3. Deep Features Model (`train_feat_model`)

**Input:** Pre-extracted feature arrays stored as `.npz` files

**Preprocessing:**
- Flattens 3D feature arrays to 2D `(n_samples, n_features)`
- Standard scaling (`StandardScaler`)
- PCA dimensionality reduction to `300` components

**Model:** LightGBM (GBDT)
- Learning rate: `0.1`
- Feature fraction: `0.3`
- Num leaves: `31`
- Rounds: `105`

---

### 4. Combined / Fused Model (`train_combined_model`)

**Input:** Concatenation of learned representations from all three branches
- 300 dims from deep features (PCA output)
- 47 dims from text sequences (padded LSTM input)
- 3 dims from emoticons (encoded features)
- Total: **350 features**

**Preprocessing:** Standard scaling on the concatenated feature matrix

**Model:** LightGBM (GBDT)
- Learning rate: `0.1`
- Feature fraction: `0.3`
- Num leaves: `31`
- Rounds: `105`

---

## Dataset Format

### Emoticon CSVs (`train_emoticon.csv`, `valid_emoticon.csv`, `test_emoticon.csv`)

| Column | Type | Description |
|---|---|---|
| `input_emoticon` | string | Sequence of Unicode emoticons (U+1F600–U+1F721) |
| `label` | int (0/1) | Binary class label *(absent in test set)* |

Example row:
```
input_emoticon,label
😀😂🙃😎🤩😋😜,1
```

### Text Sequence CSVs (`train_text_seq.csv`, `valid_text_seq.csv`, `test_text_seq.csv`)

| Column | Type | Description |
|---|---|---|
| `input_str` | string | ~50-char string with a 3-char prefix (`ID:`, `SQ:`, etc.) |
| `label` | int (0/1) | Binary class label *(absent in test set)* |

Example row:
```
input_str,label
SQ:tbmb17ovd0yn-tbbt9svd219r3j.bm47iqaf74_rts,0
```

### Feature NPZ Files (`train_feature.npz`, `valid_feature.npz`, `test_feature.npz`)

| Key | Shape | Description |
|---|---|---|
| `features` | `(n_samples, timesteps, feature_dim)` | Pre-extracted feature arrays; must flatten to ≥300 dims |
| `label` | `(n_samples,)` | Binary class labels *(absent in test set)* |

The default generated datasets use shape `(n, 10, 64)` → 640 dims after flattening, satisfying the PCA(300) requirement.

---

## Installation

```bash
pip install numpy pandas lightgbm scikit-learn tensorflow
```

Python 3.8+ is recommended. TensorFlow 2.x is required for the LSTM branch.

---

## Usage

```bash
python feature_fusion.py
```

The script reads all datasets from `datasets/`, trains all four models, and writes predictions to four `.txt` files in the working directory.

### Output Files

| File | Description |
|---|---|
| `pred_emoticon.txt` | Per-sample predictions (0 or 1) from the emoticon model |
| `pred_textseq.txt` | Per-sample predictions from the LSTM text-seq model |
| `pred_deepfeat.txt` | Per-sample predictions from the deep features model |
| `pred_combined.txt` | Per-sample predictions from the fused model |

Each file contains one integer per line corresponding to a test sample.

---

## Generating Synthetic Datasets

If you need to recreate the datasets from scratch, a dataset generator script is included. It produces all 9 files (3 splits × 3 modalities) with the correct shapes, column names, and value ranges:

```bash
python generate_datasets.py
```

Default split sizes:

| Split | Samples |
|---|---|
| Train | 800 |
| Validation | 100 |
| Test | 200 |

---

## Notes

- The emoticon model removes the **7 most frequent** characters from the training corpus before encoding. This is a dataset-dependent step — the characters removed will differ across datasets.
- The LSTM model trains for 150 epochs, which may take several minutes depending on hardware.
- The combined model uses raw learned representations (not model probabilities) as features, making it a **feature-level fusion** rather than a decision-level ensemble.
- All three sub-models must complete training before the combined model can be trained, as it depends on their output feature matrices.
