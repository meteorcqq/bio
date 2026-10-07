# PCG pretraining and quality-aware fusion

Core implementation for **Domain-Specific Self-Supervised Pretraining and Deep Quality-Aware Fusion for Cardiac Function Screening from Phonocardiograms**.

## Installation

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install matching PyTorch and torchaudio versions with a build appropriate for your CPU/CUDA environment. The initial requirements are not version-pinned; the final server environment has not yet been exported. Decoding MP3/WFDB recordings may also require an appropriate audio backend.

## Included modules

| File in `src/` | Purpose |
| --- | --- |
| `pcg_audio.py` | Mono audio/WFDB loading, resampling, normalization, and segment extraction |
| `train_ssl_byola2.py` | AudioNTT2022 encoder, paired A0--A4 augmentations, and self-supervised pretraining |
| `qc_ordinal.py` | Log-mel quality dataset, ResNet-18 CORAL/softmax models, and quality scores |
| `train_qc_ordinal.py` | Five-fold quality-model training and evaluation |
| `predict_qc_ensemble.py` | Recording-level quality scores from five saved quality-model folds |
| `attach_qc_scores.py` | Join quality scores to an outcome manifest by recording filename |
| `train_outcome.py` | Patient-level outcome training, demographic fusion, and quality ablations |
| `evaluate_outcome_checkpoint.py` | Validation/test prediction and evaluation from a saved outcome model |
| `check_byola2_compat.py` | Inspect encoder checkpoint key/shape compatibility |

## Prepare your own manifests

Use the header-only CSVs in `examples/` as schema templates. The historical field name `server_source_path` denotes the audio file path on **your** machine; it does not require access to our server. WFDB paths should identify the `.hea` file with the corresponding record files available alongside it.

| Manifest | Required fields |
| --- | --- |
| Pretraining | `dataset_id`, `server_source_path` |
| Quality training | `filename`, `server_source_path`, `quality_ordinal`, `cv_fold` |
| External quality evaluation | `filename`, `server_source_path`, `usable_high_quality` |
| Outcome | `filename`, `server_source_path`, `patient_id`, `outcome_binary`, `split` |
| Quality scores | `filename`, `quality_score_q` |

Quality labels are 0--3 in increasing quality order. `cv_fold` is 0--4; keep every subject in a single fold. Optional `start_s` and `end_s` restrict loading to a labelled interval. External binary quality labels are 0/1. Outcome labels are 0 for normal and 1 for abnormal, and `split` is `train`, `val`, or `test`; keep all recordings from one patient in a single split. Use unique recording filenames across the quality-score join.

Weighted outcome modes require `quality_score_q`. Multimodal mode additionally requires the ten numeric fields listed in `examples/outcome_manifest.csv`: sex, pregnancy, four age indicators, and four BMI indicators. The implementation consumes these encoded fields rather than imputing raw demographics. Unavailable age/BMI categories use all-zero indicators; male sex and recorded true pregnancy use one, with zero otherwise, including unavailable values.

## Example workflow

These examples assume you have filled the manifest templates with authorized data and valid local paths. They do not download data or checkpoints. Run from the repository root.

### 1. Self-supervised pretraining

```bash
python src/train_ssl_byola2.py \
  --manifest manifests/pretrain.csv --run-dir runs/ssl/a4 \
  --stage 4 --source-sampling uniform --epochs 100 --batch-size 128
```

`--stage` selects the cumulative A0--A4 augmentation configuration. `--source-sampling uniform` means equal expected probability per **source**, with uniform recording selection within each source. `none` means unweighted sampling over recordings. The local script default is `sqrt`, so specify `uniform` explicitly for source-balanced sampling. The final encoder is saved as `encoder_final.pt`.

### 2. Quality models and recording scores

Train each fold separately, using output folders `runs/qc/fold0` through `runs/qc/fold4`:

```bash
python src/train_qc_ordinal.py \
  --zch-manifest manifests/quality.csv --cdhs-manifest manifests/external_quality.csv \
  --output-dir runs/qc/fold0 --fold 0 --head coral --epochs 40

python src/predict_qc_ensemble.py \
  --manifest manifests/outcome.csv --qc-run-root runs/qc \
  --head coral --windows 1 --output runs/outcome_quality_scores.csv

python src/attach_qc_scores.py \
  --manifest manifests/outcome.csv --scores runs/outcome_quality_scores.csv \
  --output manifests/outcome_with_quality.csv
```

Repeat the training command for folds 1--4 before ensemble inference. Supply the same `--crop-seconds` to quality training and inference for your chosen configuration. The current trainer selects quality checkpoints by validation macro-F1; this initial snapshot should not be treated as a verified implementation of the manuscript's final checkpoint-selection protocol.

### 3. Outcome training and evaluation

```bash
python src/train_outcome.py \
  --manifest manifests/outcome_with_quality.csv --output-dir runs/outcome/both \
  --ssl-encoder runs/ssl/a4/encoder_final.pt --quality-mode both \
  --skip-test --epochs 60 --batch-size 32

python src/evaluate_outcome_checkpoint.py \
  --manifest manifests/outcome_with_quality.csv --checkpoint runs/outcome/both/best.pt \
  --quality-mode both --splits val test --windows 1 --output-dir runs/evaluation
```

Quality modes are `none`, `hard`, `train`, `infer`, and `both`. Add `--multimodal` consistently to training and evaluation to use the ten demographic covariates. Freeze configuration using training/validation data before requesting test evaluation; the evaluator needs validation data to determine its decision threshold. For hard filtering, supply a threshold established from your quality validation data rather than treating the snapshot's default threshold as universal. Only load checkpoints from sources you trust.

## Data access

The public outcome dataset is [CirCor / PhysioNet Challenge 2022](https://physionet.org/content/challenge-2022/1.0.0/). [ZCHSound](http://zchsound.ncrcch.org.cn/data) provides the original project portal. The extended ZCHSound quality dataset used in this study requires authorization from its provider and is not distributed here. Obtain each dataset under its own access conditions.

## Checks performed for this release

All nine Python source files pass syntax parsing. The CSV quality-score attachment is checked with synthetic success and rejection cases:

```bash
python -m unittest discover -s tests -v
```

Full training, GPU execution, and manuscript-result reproduction have not been rerun for this initial release.

## Attribution

This work builds on CardioPHON, BYOL-A/BYOL-A2, and CORAL. Please cite the corresponding papers when using these methods. The research manuscript is currently unpublished; this repository does not imply a journal acceptance or DOI.
