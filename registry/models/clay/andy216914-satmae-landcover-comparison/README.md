# Clay v1.5 vs. SatMAE on Sentinel-2 land cover (EuroSAT)

**Proof-of-concept experiment (VIP Assignment 5), not a production recipe.**
It builds on the verified Clay inference recipe in
[`../andy216914-embedding-inference/`](../andy216914-embedding-inference/),
whose environment, checkpoint and model loader it reuses.

## 1. Question

Clay is a general-purpose embedding model meant to be used frozen, for any
sensor. SatMAE is a masked autoencoder trained only on Sentinel-2. This
experiment asks:

1. Do frozen Clay v1.5 embeddings support Sentinel-2 land-cover
   classification, how do they compare with SatMAE's under an identical
   protocol, and do the two carry complementary information?
2. The two models' official EuroSAT code disagree on the band order inside
   EuroSAT's files. Which order is correct, and what does the wrong one cost?

Neither model was trained on EuroSAT.

## 2. Design

| Element | Choice |
|---|---|
| Data | EuroSAT, all 13 Sentinel-2 bands: 27,000 images of 64 × 64 px, 10 classes. TorchGeo's copy, pinned revision, sha256-checked |
| Split | The standard split files (train 16,200, validation 5,400, test 5,400), sha256-checked |
| Band order | Measured from the data (`scripts/prepare_eurosat.py`): the cirrus band B10 is nearly zero, so its position identifies the layout; B8A's correlation with B08 confirms it. Written to `results/band_order.json` |
| Shared input | All images packed into one array file that both models read |
| Clay | v1.5 checkpoint, loaded with the inference recipe's `load_encoder`. Clay's own EuroSAT settings (commit `f14e698`): bands B02–B08, B8A, B11, B12; sentinel-2-l2a normalization; their wavelengths; 10 m; zeros for time and location; 64 × 64 input. Feature: class token |
| SatMAE | Pretrained group-channel ViT-Large (`pretrain-vit-large-e199.pth`, Zenodo 7338613), code at `0b210ace`. SatMAE's own EuroSAT evaluation transform: normalize, resize to 109, center-crop 96, drop B01/B09/B10, three band groups. Feature: average of the patch tokens. Loading is checked: only one zero-initialized tensor may be missing |
| Probe | Standardized features, logistic regression; C ∈ {0.01, 0.1, 1, 10} chosen on validation; test split used once |
| Compared | Majority class; spectral statistics (mean and std of the 10 bands); Clay; SatMAE; Clay + SatMAE concatenated. Secondary: the other pooling of each model. Band order: SatMAE with the order its published code assumes |
| Reported | Test accuracy with 95% bootstrap intervals, macro-F1, per-class accuracy, confusion matrices, exact McNemar tests on paired predictions |

## 3. Layout

```
config.sh          paths, pinned URLs and checksums; sources the Clay recipe's settings
run_all.sh         one command: checks prerequisites, picks a GPU, submits jobs 1-4
env/               SatMAE environment (Python 3.9, torch 1.12.1, timm 0.3.2)
jobs/              1_prepare_data, 2_setup_satmae (CPU), 3_extract_features (GPU), 4_linear_probe (CPU)
scripts/           prepare_eurosat.py, embed_clay.py, embed_satmae.py, linear_probe.py
results/           band_order.json, metrics.json, results_table.md, test_predictions.csv, figures/, logs/
```

Large files (dataset, packed arrays, SatMAE environment and checkpoint,
features) stay in `$CLAY_WORKDIR/satmae-landcover/` on scratch, about 15 GB.

## 4. Running it

Requirements: a PACE ICE account with the Slurm settings of the Clay recipe
(Section 5.3 of its README).

```bash
git clone --branch clay-satmae-landcover-comparison https://github.com/Andy216914/cybershuttle-ai4science-foundry.git ~/clay-poc
cd ~/clay-poc/registry/models/clay/andy216914-satmae-landcover-comparison
bash run_all.sh            # submits the jobs and returns
squeue -u $USER            # repeat until no poc-* jobs remain
cat results/results_table.md
```

`run_all.sh` reuses the Clay environment and checkpoint when they exist for
`$CLAY_WORKDIR` (default `~/scratch/clay`); otherwise it first submits the
Clay recipe's own setup jobs. It picks whichever of L40S, A100 or V100
Slurm expects to start first (`POC_GPU=a100 bash run_all.sh` to choose).

| Job | Resources | Purpose |
|---|---|---|
| 1 `prepare_data` | CPU, 4 cores, 32 GB | download and check EuroSAT, determine band order, pack arrays |
| 2 `setup_satmae` | CPU, 4 cores, 16 GB | SatMAE code, environment, checkpoint (download limited by Zenodo, ~20 min) |
| 3 `extract_features` | 1 GPU, 48 GB | Clay and SatMAE features for all 27,000 images |
| 4 `linear_probe` | CPU, 8 cores, 32 GB | probes, metrics, figures |

Each job logs to `results/logs/<job>-<id>.out`; a successful job ends with
`... COMPLETED`.

## 5. Results

Filled in from `results/` after the run.

## 6. Limitations

- EuroSAT is Sentinel-2 Level-1C (top of atmosphere); Clay's normalization
  statistics are for Level-2A. This follows Clay's own EuroSAT code.
- Clay receives zeros for time and location, as in Clay's EuroSAT code,
  although the GeoTIFFs are georeferenced.
- SatMAE sees images upsampled from 64 to 96 px; Clay sees them at 64 px.
- Frozen features and a linear probe only; one standard split.

## 7. Credits

SatMAE environment versions and the timm 0.3.2 patch follow DevP1130's
SatMAE recipe (PR #28). EuroSAT: Helber et al., 2019. SatMAE: Cong et al.,
NeurIPS 2022. Clay: Clay Foundation, v1.5.
