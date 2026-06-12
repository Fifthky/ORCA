# ORCA: Blackbox Time-Series Forecasting Refinement Framework

A unified evaluation framework for blackbox time-series forecasting model refinement. This project runs end-to-end experiments on CSV datasets, comparing multiple refiner architectures across frozen foundation models.

**Scope**:
- CSV datasets only (no external data loaders)
- Frozen base models (no gradient backpropagation to base model)
- Online and batch refiner training modes
- Reproducible summary CSVs with per-dataset, per-model metrics

---

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies**:
- PyTorch 2.4.0+
- GluonTS (evaluation metrics)
- transformers 4.40.1+ (model backends)
- chronos-forecasting, timesfm, uni2ts (foundation model packages)

---

## 2. Dataset Preparation

### 2.1 Supported Datasets

8 benchmark time-series datasets:
ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather, Traffic, Electricity
### 2.2 Download Datasets

Download all datasets:
```bash
python -m data.download_CSV --dataset all
```

Download single dataset:
```bash
python -m data.download_CSV --dataset ETTh1
```

Cached files are stored in `data/data_cache/`. Use `--force` to re-download.

### 2.3 Use Custom CSV

Provide any CSV file directly:
```bash
python run.py --csv_path /path/to/your/data.csv --target_column all
```

- `--target_column all`: multivariate (all numeric columns)
- `--target_column OT`: single column (or fallback to last column)

---

## 3. Base Models

### 3.1 Default Models (run with `--model all`)

| Short Name | Full Name | Default Context |
|------------|-----------|-----------------|
| `chronos-2` | Chronos-2 | 520 |
| `moirai-2` | Moirai-2.0-R-small | 520 |
| `tirex` | TiRex-1.1-gifteval | 520 |
| `timesfm-2` | TimesFM-2.5-200m | 520 |
| `sundial` | Sundial-base-128m | 520 |

### 3.2 Additional Models (not in default `all`)

- `moirai-1-small`, `moirai-1-base`, `moirai-1-large`

To use them, explicitly specify:
```bash
python run.py --model moirai-1-large --dataset ETTh1
```

### 3.3 Model Checkpoints

Default path: `path_to_your_model`

Override with `--tsfm_model_prefix`:
```bash
python run.py --tsfm_model_prefix /your/path/to/models
```

Or specify exact path for single model:
```bash
python run.py --model chronos-2 --tsfm_local_path /path/to/chronos-2
```

---

## 4. Refiners (Core Components)

9 refiner architectures, categorized into two groups:

### 4.1 Our Proposed Refiners (Linear/Bay/Attn/Bay_Attn)

Support full ablation study with configurable parameters:

| Refiner | Description | Training Modes |
|---------|-------------|----------------|
| `Linear` | Decomposition + linear mapper + channel mixer | batch, online |
| `Bay` | Bayesian Decay buffer with prior loss | online only |
| `Attn` | Attention-based refiner | batch, online |
| `Bay_Attn` | Bayesian + attention | online only |

**Configurable dimensions**:

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--refiner_input` | `all`, `xy`, `x`, `y`, `e_past` | `all` | Input covariates to refiner |
| `--update_rule` | `plain`, `bayesian` | `plain` | Loss function for training |
| `--training_method` | `batch`, `online` | `online` | Training protocol |
| `--online_buffer_windows` | int (e.g. 512, 3000, 6000) | `3000` | Buffer size before update trigger |
| `--batch` | int | `256` | Bay/Bay_Attn training batch size |
| `--bay_loss` | `mse`, `mae`, `huber` | `mse` | Bay refiner observation loss |
| `--force_gate_open` | flag | `False` | Disable confidence gate (ablation) |
| `--channel_mix` / `--no-channel_mix` | flag | `True` | Enable/disable channel mixing |

**Input mode semantics** (`--refiner_input`):

| Mode | Inputs | Description |
|------|--------|-------------|
| `xy` | Base Model Input + Base Model Prediction | **OrCA (proposed)** |
| `all` | x + y + Error Signal | All three covariates |
| `x` | Base Model Input (history) | Only historical context |
| `y` | Base Model Prediction | Only base model output |
| `e_past` | Error Signal | Only past prediction error |

### 4.2 Baseline Refiners (AdaY/DSOF/TAFAS/SOLID/ELF)

External methods integrated for comparison. Limited configurability:

| Refiner | Special Parameters |
|---------|-------------------|
| `AdaY` |`--baseline_router` |
| `DSOF` | `--baseline_router` |
| `TAFAS` | `--baseline_router` |
| `SOLID` | `--baseline_router`, `--solid_period` |
| `ELF` | `--baseline_router` |

**Router mode** (`--baseline_router`):
- **Disabled (default)**: Vanilla baseline refiner
- **Enabled**: Activates Boltzmann router for dynamic selection between base model and refiner output

---

## 5. Running Experiments

### 5.1 Main Experiment (Proposed Method)

Bayesian refiner, OrCA input (xy), bayesian update, buffer=3000, batch=256:

```bash
python run.py \
  --dataset all \
  --model all \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian \
  --online_buffer_windows 3000 \
  --batch 256 \
  --pred_len 96 \
  --context_length 520 \
  --device cuda
```

**This is the core configuration for the proposed method (OrCA).**

### 5.2 Input Ablation Study

Test all 5 input variants:

```bash
python run.py \
  --dataset all \
  --model all \
  --refiner bay \
  --refiner_input xy all x y e_past \
  --update_rule bayesian \
  --online_buffer_windows 3000 \
  --batch 256 \
  --pred_len 96
```

This runs 5 experiments (one per input mode), results saved separately.

### 5.3 Update Rule Ablation

Compare plain vs bayesian update:

```bash
python run.py \
  --refiner bay \
  --refiner_input xy \
  --update_rule plain bayesian \
  --online_buffer_windows 3000 \
  --batch 256
```

### 5.4 Buffer Size Study

```bash
python run.py \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian \
  --online_buffer_windows 2000 3000 4000 \
  --batch 256
```

### 5.5 Batch Size Study (Bay Refiner)

```bash
python run.py \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian \
  --online_buffer_windows 3000 \
  --batch 128 256 512
```

### 5.6 Baseline Refiners (with Router)

```bash
# Vanilla baseline
python run.py --refiner dsof --dataset ETTh1 --model chronos-2

# With Boltzmann router
python run.py --refiner dsof --dataset ETTh1 --model chronos-2 --baseline_router
```

### 5.7 Control Group (No Refiner)

Baseline metrics are automatically computed for every run (before refiner is applied). Results include both `baseline` and `refined` rows in output CSVs.

### 5.8 Multi-Prediction Length

```bash
python run.py \
  --pred_len 30 96 336 \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian
```

Automatically generates per-pred_len CSVs + one `pred_avg` average file.

### 5.9 Single Dataset + Single Model

```bash
python run.py \
  --dataset ETTh1 \
  --model moirai-2 \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian \
  --online_buffer_windows 3000 \
  --batch 256 \
  --pred_len 96
```

### 5.10 Resume Interrupted Evaluation

```bash
python run.py \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian \
  --resume_eval
```

Skips completed (non-NaN) entries, only runs missing/failed combinations.

---

## 6. Parameter Lists (Space-Separated)

The following arguments accept **multiple values** (space-separated), automatically expanding to run all combinations:

| Argument | Example | Runs |
|----------|---------|------|
| `--dataset` | `ETTh1 ETTh2 Weather` | 3 datasets |
| `--model` | `chronos-2 moirai-2` | 2 models |
| `--refiner` | `bay linear adaY` | 3 refiners |
| `--refiner_input` | `xy all x` | 3 input modes |
| `--update_rule` | `plain bayesian` | 2 update rules |
| `--training_method` | `batch online` | 2 training modes |
| `--online_buffer_windows` | `2000 3000 4000` | 3 buffer sizes |
| `--bay_loss` | `mse mae huber` | 3 loss variants |
| `--pred_len` | `30 96 336` | 3 prediction lengths |

**Example**: `--refiner_input xy all --update_rule plain bayesian` → 2×2=4 variants.

---

## 7. Output Layout

### 7.1 Detailed Per-Dataset CSVs

`results/details/single_dataset/<dataset>/results_csv_<dataset>_<model>_<Refiner>_<suffix>.csv`

Contains `baseline` and `refined` rows with MAE/MSE/MAE_raw/MSE_raw.

### 7.2 Summary CSVs (Aggregated)

`results/results_csv_<Refiner>_<suffix>_pred<Len>_all_csv_dataset_mse.csv`  
`results/results_csv_<Refiner>_<suffix>_pred<Len>_all_csv_dataset_mae.csv`

Format:
- Rows: datasets (ETTh1, ETTh2, ...) + `Models Avg.`
- Columns: each base model (Chronos-2, Moirai-2, ...) with `Value` and `Change` (%)
- `Change` = (refined - baseline) / baseline × 100 (negative = improvement)

**Suffix examples**:
- `Bay_xy_bayesian_buf3000_batch256` → proposed method
- `Bay_xy_plain_buf3000_batch256` → plain update ablation
- `Linear` → baseline refiner (no suffix)
- `router` → baseline refiner with router enabled

### 7.3 Prediction Length Average

When `--pred_len` has multiple values:
- `..._pred_avg_all_csv_dataset_mse.csv` → average across all pred_lens

### 7.4 Logs and Plots

- `results/details/logs/` → training logs, loss curves (JSON + PNG)
- `results/details/logs/online_training_runs/` → online training specific logs
- `results/plots/` → comparison plots
- `results/attn_maps/` → attention map visualizations (if `--attn_maps` enabled)

### 7.5 Inference Cache

`data/model_infer_cache/` → cached base model predictions (reuse with `--cache`)

---

## 8. Advanced Features

### 8.1 Attention Map Inspection

```bash
python run.py --model moirai-2 --dataset ETTh1 --attn_maps
```

Extracts and saves attention weights from supported models (moirai-2, chronos-2).

### 8.2 Channel-Wise Normalization (Electricity)

Automatically enabled for `Electricity` dataset:
- Metrics: computed per-channel, then averaged
- Refiner input: normalized per-channel (not global)

Other datasets use global normalization.

### 8.3 Large Data Channel Mode

Automatically activated for `{Traffic, Electricity}` when `pred_len > 100`:
- Reduces memory footprint during evaluation
- Processes channels sequentially for single-channel backends

### 8.4 Bay Loss Variants

Only for `Bay` refiner:
```bash
python run.py --refiner bay --bay_loss mse      # L2 loss (default)
python run.py --refiner bay --bay_loss mae      # L1 loss
python run.py --refiner bay --bay_loss huber    # Huber loss (delta=1.0)
python run.py --refiner bay --bay_loss huber --bay_huber_delta 0.5  # custom delta
```

### 8.5 Ablation Flags

```bash
# Disable confidence gate (force c_t = 1.0)
python run.py --refiner bay --force_gate_open

# Disable channel mixing (CI-style channel-independent)
python run.py --refiner bay --no-channel_mix
```

---

## 9. Plotting Scripts

### 9.1 Structure Ablation

```bash
python plot/plot_abla_structure.py
```

Plots 5 variants: proposed, w/o router, w/o channel mixing, w/o bayesian prior, w/o decay buffer.

### 9.2 Input Ablation

```bash
python plot/plot_abla_input.py
```

Horizontal stacked bar chart: 6 datasets × 5 input modes (xy, all, x, y, e_past).

### 9.3 Hyperparameter Heatmap

```bash
python plot/plot_hyper.py 
```

Heatmap: buffer size (cols) × batch size (rows) × 5 models.

---

## 10. Quick Reference

### Standard Proposed Method Run
```bash
python run.py \
  --dataset all \
  --model all \
  --refiner bay \
  --refiner_input xy \
  --update_rule bayesian \
  --online_buffer_windows 3000 \
  --batch 256 \
  --pred_len 96 \
  --context_length 520 \
  --device cuda
```

### Compare All Input Modes
```bash
python run.py \
  --dataset all \
  --model all \
  --refiner bay \
  --refiner_input xy all x y e_past \
  --update_rule bayesian \
  --online_buffer_windows 3000 \
  --batch 256 \
  --pred_len 96
```

### Baseline with Router
```bash
python run.py \
  --refiner dsof ta fas solid elf \
  --baseline_router \
  --dataset all \
  --model all \
  --pred_len 96
```

### Resume After Interruption
```bash
python run.py --refiner bay --refiner_input xy --update_rule bayesian --resume_eval
```

---

## 11. Notes

- `timesfm-2` uses the `timesfm` Python package from active environment
- All refiners are **blackbox**: base model runs once, no gradient backpropagation
- Default context length: 520 
- Default prediction length: 96
