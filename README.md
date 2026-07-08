# LS-CATNet: Lightweight Cross-Scale Attention Transformer Network for Wildfire Smoke Detection

This repository provides a reproducible PyTorch implementation of **LS-CATNet** for four-class wildfire smoke-scene classification in remote-sensing images.

The implementation follows the model used for the manuscript results:

- shallow EfficientNet-B0 feature extractor with partial freezing;
- four multi-scale patch-token branches with patch sizes 5, 7, 9 and 14;
- separable self-attention transformer blocks;
- cross-scale coordinate-attention bridges;
- adaptive branch-wise softmax-gated fusion;
- four-class classifier.

---

## 1. Repository structure

```text
LS-CATNet/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── configs/
│   ├── iiitdmj.yaml
│   └── ustc.yaml
├── scripts/
│   ├── prepare_splits.py
│   └── audit_split_integrity.py
└── src/
    └── lscatnet/
        ├── __init__.py
        ├── model.py
        ├── data.py
        ├── metrics.py
        ├── plots.py
        ├── utils.py
        ├── train.py
        ├── evaluate.py
        ├── benchmark.py
        ├── count_params.py
        └── predict.py
```

---

## 2. Installation

```bash
git clone <repository-url>
cd LS-CATNet
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
pip install -e .
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate lscatnet
pip install -e .
```

---

## 3. Expected dataset structure

The training/evaluation scripts expect already split folders:

```text
DATA_ROOT/
├── train/
│   ├── fognonsmoke/
│   ├── fogsmoke/
│   ├── nonsmoke/
│   └── smoke/
├── val/
│   ├── fognonsmoke/
│   ├── fogsmoke/
│   ├── nonsmoke/
│   └── smoke/
└── test/
    ├── fognonsmoke/
    ├── fogsmoke/
    ├── nonsmoke/
    └── smoke/
```

The default class order is:

```text
fognonsmoke, fogsmoke, nonsmoke, smoke
```

Use `--class-names` to change the order or folder names.

---

## 4. Leakage-safe split and augmentation protocol

The repository implements a leakage-safe dataset preparation protocol in which the train/validation/test split is performed before any augmentation.

```text
raw, unaugmented images
        ↓
train/validation/test split at original-image level
        ↓
offline augmentation only for train images, if needed
        ↓
validation and test remain independent and unaugmented
```

Run:

```bash
python scripts/prepare_splits.py \
  --raw-root /path/to/raw_unaugmented_dataset \
  --output-root /path/to/split_dataset \
  --class-names fognonsmoke fogsmoke nonsmoke smoke \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42 \
  --copy-originals \
  --augment-train \
  --augment-factor 4
```

This script writes `split_manifest.csv`, which records the original image ID, split, class and saved file path. The `orig_id` field proves that augmented variants of one original image are confined to the training split.

Audit the split:

```bash
python scripts/audit_split_integrity.py \
  --data-root /path/to/split_dataset \
  --manifest /path/to/split_dataset/split_manifest.csv \
  --hash-audit
```

Expected successful output:

```text
Manifest audit passed: all original images are confined to one split each.
Exact-hash audit passed: no identical image files occur across different splits.
```

---

## 5. Count parameters

```bash
python -m lscatnet.count_params --no-pretrained
```

The manuscript-scale model should report approximately:

```text
Total parameters      : 515,802  ≈ 0.516 M
Trainable parameters  : 513,426  ≈ 0.513 M
```

The exact value can vary only if the architecture, backbone truncation or fusion module is changed.

---

## 6. Training

```bash
python -m lscatnet.train \
  --data-root /path/to/split_dataset \
  --outdir outputs_lscatnet_ustc \
  --class-names fognonsmoke fogsmoke nonsmoke smoke \
  --num-classes 4 \
  --img-size 224 \
  --batch-size 16 \
  --epochs 60 \
  --lr 1e-4 \
  --eta-min 1e-6 \
  --patience 10 \
  --weight-decay 0.0
```

The training script uses:

- optimizer: Adam;
- loss: standard Cross-Entropy by default;
- scheduler: CosineAnnealingLR by default;
- online augmentation only in the training data loader;
- validation and test transforms: resize, tensor conversion and ImageNet normalization only.

For class-weighted cross-entropy:

```bash
python -m lscatnet.train \
  --data-root /path/to/split_dataset \
  --outdir outputs_weighted \
  --weighted-loss
```

---

## 7. Evaluation

```bash
python -m lscatnet.evaluate \
  --data-root /path/to/split_dataset \
  --checkpoint outputs_lscatnet_ustc/LSCATNet_best.pth \
  --outdir outputs_lscatnet_ustc_eval \
  --class-names fognonsmoke fogsmoke nonsmoke smoke \
  --bootstrap 1000
```

Outputs include:

```text
metrics.json
per_class_metrics.csv
predictions.csv
confusion_matrix.png
roc_curves.png
```

The `--bootstrap` option computes 95% confidence intervals for accuracy, macro-F1, macro-FAR and macro-AUC. The --bootstrap option computes 95% confidence intervals for the reported performance metrics, providing statistical uncertainty estimates.

---

## 8. Inference-speed and FLOP benchmark

```bash
python -m lscatnet.benchmark \
  --checkpoint outputs_lscatnet_ustc/LSCATNet_best.pth \
  --device cuda \
  --runs 200 \
  --warmup 50 \
  --out benchmark_gpu.json
```

The output JSON includes:

- device name;
- mean latency;
- standard deviation;
- FPS;
- GMACs;
- GFLOPs estimated as `2 × MACs`;
- parameter count.

For edge-device reporting, run the same command on the target device, such as an NVIDIA Jetson module, and include the resulting JSON in the repository.

---

## 9. Single-image prediction

```bash
python -m lscatnet.predict \
  --image /path/to/image.jpg \
  --checkpoint outputs_lscatnet_ustc/LSCATNet_best.pth \
  --class-names fognonsmoke fogsmoke nonsmoke smoke
```

## 12. Citation

Please cite the manuscript if using this code.
