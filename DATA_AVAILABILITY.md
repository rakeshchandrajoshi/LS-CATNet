# Data and Code Availability Statement Template

The source code, model configuration files, dataset-preparation scripts, split-integrity audit scripts, training scripts, evaluation scripts, benchmark scripts and reproducibility instructions for LS-CATNet are available at:

**GitHub:** `https://github.com/rakeshchandrajoshi/LS-CATNet/`  
**DATASET / DOI:** `(https://data.mendeley.com/datasets/4mn2g8cnsf/1)`

The repository includes:

1. the LS-CATNet PyTorch implementation;
2. the leakage-safe dataset splitting script;
3. the train-only offline augmentation workflow;
4. split manifests identifying the original-image-level train/validation/test partitions;
5. the training and evaluation scripts used to reproduce the reported metrics;
6. the parameter-count and FLOP/latency benchmarking scripts;
7. generated result tables and configuration files.

The IIITDMJ Smoke and USTC SmokeRS datasets are available from their original data providers. Due to dataset licensing conditions, raw image files are not redistributed in this repository unless permitted by the original dataset license. The split manifests and reproducibility scripts are provided to allow exact regeneration of the experimental protocol.
