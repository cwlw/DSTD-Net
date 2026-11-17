# DSTD‑Net: A Dual‑Stream Temporal‑Difference Network for Cross‑Subject Continuous EEG Emotion Regression

This repository is the official PyTorch implementation of the paper:
**"DSTD‑Net: A Dual‑Stream Temporal‑Difference Network for Cross‑Subject Continuous EEG Emotion Regression"** **(Journal/Conference decision pending)**.

## 📖 Introduction

Cross‑subject continuous EEG emotion regression faces three core challenges: complementary spectral features cannot be fully exploited, temporal mixer capacity mismatches the small‑sample leave‑one‑subject‑out(LOSO) setting, and point‑wise loss fails to constrain temporal dynamics of affective trajectories. We propose **DSTD‑Net**, addressing these three limitations based on the EmT baseline:

1. **Dual‑stream RMPG Encoder**:Two parameter‑independent Residual Multiview Pyramid Graph (RMPG) encoders separately process rPSD and differential entropy(DE) features, whose output token sequences are fused by a single‑parameter learnable scalar gate α.
2. **Capacity‑matched lightweight temporal mixer**: Adopts one‑layer bidirectional GRU inside Temporal Contextual Transformer(TCT), instead of two‑layer Bi‑GRU, to reduce model variance and avoid subject‑level overfitting under LOSO cross‑validation.
3. **Four‑term temporal‑difference training objective**: Combines CCC loss, MSE loss, temporal‑difference loss L<sub>TD</sub> and smoothness regularizer. The temporal‑difference loss explicitly supervises frame‑to‑frame changes of prediction trajectory. Shared channel masking is further applied during training as regularization.

## ⚙️ Environment Setup

We recommend using a Conda environment. The code has been tested with Python 3.10.8 and PyTorch 2.1.2.

```bash
# Create virtual environment
conda create -n dslemt python=3.10.8
conda activate dslemt

# Install dependencies via pip
pip install -r requirements.txt
```

*Note: Please refer to the `requirements.txt` file in the repository for specific library versions, including `einops`, `tqdm`, `scikit-learn`, and `pandas`.*

## 📁 Repository Structure

```text
.
├── base/                       # Core functional modules
│   ├── checkpointer.py         # Model saving and checkpoint recovery
│   ├── dataset.py              # Data loading, windowing, and LOSO partitioning
│   ├── eeg.py                  # EEG signal processing helpers (Note: historical filename in repo)
│   ├── experiment.py           # Base experiment class defining general workflows
│   ├── feature_fusion.py       # Early feature fusion module, deprecated; token‑level learnable α‑gate is implemented inside model/EmT.py
│   ├── label_configs.py        # Label dimension mapping configurations
│   ├── logger.py               # Metric calculation (RMSE, PCC, CCC) and logging
│   ├── loss_function.py        # Hybrid losses (CCC, MSE, Temporal Difference, Smoothness)
│   ├── parameter_control.py    # Optimizer and learning rate control
│   ├── preprocessing.py        # Raw EEG filtering and preprocessing base class
│   ├── scheduler.py            # Learning rate scheduler (Warmup + ReduceLROnPlateau)
│   ├── trainer.py              # Training/testing loops with channel masking logic
│   └── utils.py                # General utilities (pickle IO, file management)
├── model/                      # Model architecture
│   └── EmT.py                  # Core network: Dual-stream RMPG, learnable α gate, TCT, 1-layer Bi-GRU
├── main.py                     # Main entry point, parses CLI args and starts experiments
├── experiment.py               # Inherits from base, assembles Model, Dataset, and Trainer
├── configs.py                  # Global configuration (hyperparameters, paths, frequency bands)
├── generate_dataset.py         # Offline preprocessing script for MAHNOB-HCI
├── generate_result_csv.py      # Script to aggregate MAHNOB 24-fold LOSO results
│
├── prepare_deap.py             # [DEAP Extension] Feature extraction script for DEAP
├── fix_deap_stats.py           # [DEAP Extension] Calculates global normalization stats for DEAP
├── calc_deap_acc.py            # [DEAP Extension] Calculates Trial-level binary accuracy on DEAP
├── test_deap_only.py           # [DEAP Extension] Pure inference script for quick DEAP validation
├── check_label.py              # Helper script to inspect label shapes and distributions
└── README.md                   # Project documentation
```

## ⚠️ Guide for DEAP Extension

**Important**: The core scripts in this repository (e.g., `dataset.py`, `trainer.py`, `experiment.py`) are configured by default for **MAHNOB‑HCI (Continuous Trajectory Regression)**.

⚠️ DEAP only provides one discrete valence rating for each trial, which requires a `Sequence‑to‑One` regression protocol.
> Note: The temporal‑difference loss L<sub>TD</sub> and smoothness regularizer cannot be used for DEAP, since there are no frame‑wise continuous annotations. You **must** pass argument `-use_temporal 0` during DEAP training.

To reproduce the DEAP transfer experiments reported in our paper, apply the following source‑code modifications:

1. **`configs.py`**
   * Change `"sampling_frequency"` inside `eeg_config` from `256` to `128`.
   * Update `"output_root_directory"` to your local DEAP output directory.

2. **`base/dataset.py`**
   * Modify `DataArranger.partition_range_fn()`: Replace the hard‑coded 24‑fold split logic, dynamically generate 32 LOSO folds according to `subject_no`.
   * Modify `_EEGFeatureMixin.get_example()`: Average labels across temporal dimension with `np.mean(example, axis=0)`, compressing frame‑wise labels into a single scalar value with shape `(1,)`.

3. **`model/EmT.py`**
   * In `forward()` function, after the `self.transformer` block, add global temporal pooling:
     `fused_pooled = torch.mean(fused, dim=1)`
     This aggregates the 96‑frame token sequence into one feature vector before feeding into the regression MLP head.

4. **`base/trainer.py`**
   * Inside training loop `loop()`, remove the original continuous‑label dimension validation logic.
   * After obtaining `outputs = self.model(...)`, expand dimensions for both outputs and labels:
     `unsqueeze(1).repeat(1, seq_len, 1)`
     This adapts outputs to the 3‑dimensional input requirement of metric functions in `logger.py`.

5. **`generate_result_csv.py`**
   * Change `num_folds=24` to `num_folds=32` for aggregating results over DEAP’s 32‑fold LOSO evaluation.


*(Note: When running DEAP training, you **must** add the parameter `-use_temporal 0` to disable the temporal difference loss, as trial-level labels do not have adjacent-frame variations).*

## 🚀 Training & Evaluation

### 1. Data Preparation

* **MAHNOB-HCI**: Follow the EmT benchmark protocol. Place your preprocessed `.npy` features in a directory and update the paths in `configs.py`.
* **DEAP**: Use the `prepare_deap.py` and `fix_deap_stats.py` scripts to generate features and stats.

### 2. Reproducing MAHNOB-HCI Main Results (Table I)

To run the 24‑fold LOSO cross‑validation for the full **DSTD‑Net** model (MAHNOB‑HCI continuous valence regression):

```bash
python main.py \
  -experiment_name DSTD_Net_MAHNOB \
  -stamp MAHNOB_A0 \
  -gpu 0 \
  -case loso \
  -task reg \
  -num‑eeg‑chan 32 \
  -num‑f 6 \
  -window_length 96 \
  -hop_length 32 \
  -batch_size 2 \
  -mse_weight 2.0 \
  -use_temporal 1 \          # 开启时序差分损失（MAHNOB‑HCI连续标签启用）
  -temporal_weight 0.05 \    # 时序差分损失权重 λ₂=0.05
  -dataset_path [YOUR_MAHNOB_OUT_PATH]
```

### 3. Reproducing DEAP Extension Results

*(Remember to follow the "Guide for DEAP Extension" above and disable temporal loss)*

```bash
python main.py \
  -experiment_name DSTD_Net_DEAP \
  -stamp DEAP_TrialReg \
  -gpu 0 -case loso -task reg \
  -num‑eeg‑chan 32 -num‑f 6 \
  -window_length 96 -hop_length 32 \
  -batch_size 2 \
  -mse_weight 2.0 \
  -use_temporal 0 \         # DEAP单试次标签，关闭时序差分损失
  -dataset_path [YOUR_DEAP_OUT_PATH]
```

### 4. Aggregating Results

After training, use the `generate_result_csv.py` script to aggregate the fold results into a single `result.csv`:

```bash
python generate_result_csv.py -path_to_read_result [YOUR_SAVE_PATH]
```

## 📦 Pre-trained Models & Weights

We provide checkpoint weights and training logs for full **DSTD‑Net** on MAHNOB‑HCI (all 24 LOSO folds).

* **Baidu Netdisk**: [https://pan.baidu.com/s/1MBAx0onlvE8oyNH9Qlbj6w?pwd=pp2s](https://pan.baidu.com/s/1MBAx0onlvE8oyNH9Qlbj6w?pwd=pp2s) 
* **Extraction Code**: `pp2s`

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{yourkey_dstdnet,
  title={DSTD‑Net: A Dual‑Stream Temporal‑Difference Network for Cross‑Subject Continuous EEG Emotion Regression},
  author={Author 1, Author 2, Author 3},
  journal={Under Review},
  year={2026}
}
```

## 🙏 Acknowledgements

This codebase is heavily built upon the open‑source EmT repository [GitHub](https://github.com/yi-ding-cs/EmT). We sincerely thank the authors for releasing their valuable code and benchmark protocol for cross‑subject EEG emotion analysis.

## 📬 Contact

For any questions or bug reports, please contact: 3327264882@qq.com

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
