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
features) stay in `$CLAY_WORKDIR/satmae-landcover/` on scratch.

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
| 4 `linear_probe` | CPU, 24 cores, 64 GB | probes (28 fits in parallel), metrics, figures; about 2 min |

Each job logs to `results/logs/<job>-<id>.out`; a successful job ends with
`... COMPLETED`.

## 5. Results

One run on PACE ICE, 2026-11-22: jobs 1–3 at commit `2d29fa7`, job 4 at
`d70ff58` (the parallel probe). The logs of jobs 1–4 are in `results/logs/`
(the Clay setup jobs, when submitted, log into the inference recipe's folder).

**Band order** (`results/band_order.json`). The darkest band (mean 12 DN, the
cirrus band B10) is at file index 9, and the band at index 12 correlates 0.998
with B08. So TorchGeo's copy stores B8A last, as TorchGeo says, and Clay's
EuroSAT code reads it correctly. SatMAE's EuroSAT code assumes B8A is ninth
(its normalization constants put the near-zero cirrus mean at index 10). On
this copy it therefore feeds B09 where B8A belongs, B12 where B11 belongs and
B8A where B12 belongs, and never sees B11. Nothing fails or warns.

**Linear probes** (test split, 5,400 images; `results/results_table.md`, `results/metrics.json`)

| Features | Dim | Best C | Test accuracy [95% CI] | Macro-F1 |
|---|---|---|---|---|
| Majority class | – | – | 0.103 [0.095, 0.111] | 0.019 |
| Spectral statistics | 20 | 10 | 0.876 [0.868, 0.885] | 0.871 |
| **Clay v1.5** (class token) | 1,024 | 0.01 | **0.978** [0.974, 0.982] | 0.977 |
| **SatMAE** (patch average) | 1,024 | 10 | **0.944** [0.938, 0.950] | 0.942 |
| Clay + SatMAE | 2,048 | 0.01 | 0.980 [0.976, 0.984] | 0.979 |
| Clay, patch average | 1,024 | 0.1 | 0.977 [0.973, 0.981] | 0.976 |
| SatMAE, class token | 1,024 | 1 | 0.946 [0.940, 0.952] | 0.943 |
| SatMAE, published band order | 1,024 | 1 | 0.951 [0.945, 0.956] | 0.948 |

Paired McNemar tests on the same test images (images only A gets right vs.
only B): Clay vs. SatMAE 241 vs. 59 (p = 3 × 10⁻²⁷); Clay + SatMAE vs. Clay
26 vs. 15 (p = 0.12); SatMAE, correct vs. published band order 122 vs. 159
(p = 0.03, not significant after a Bonferroni correction for the five tests);
each model vs. spectral statistics p < 10⁻⁴⁸. Every selected model converged.

**Observations**

1. Frozen Clay embeddings transfer to Sentinel-2 land cover: 97.8% test
   accuracy, 10 points above the spectral baseline, although Clay never saw
   EuroSAT and received no date or location.
2. Clay beats SatMAE by 3.4 points, and in every class; the largest gaps are
   Highway (0.946 vs. 0.861), herbaceous vegetation (0.967 vs. 0.921) and
   river (0.977 vs. 0.934). The gap holds for
   both pooling choices, so it is not about which token is used. A plausible
   reason, not tested here: Clay's training also matches DINOv2 features,
   which tends to give embeddings that work with a linear classifier, while
   SatMAE is a plain masked autoencoder usually fine-tuned before use.
3. Joining the two adds 0.2 points, which is not significant: at this level
   SatMAE adds little that Clay does not already have.
4. The band-order error is silent. With the published order SatMAE is not
   worse (0.7 points higher, not significant after correction), so accuracy
   cannot reveal this kind of preprocessing error; checking the data can.

**Run record.** GPU: Tesla V100-PCIE-16GB. Clay took 99 s for 27,000 images
(2.97 GB peak GPU memory), SatMAE 723 s and 822 s for the two band orders; job
3 took 30 min 33 s in total. Job 4 first fitted the probes one after another
on 8 CPUs and was cancelled after 24 minutes with 3 of 8 feature sets done (log
kept, `poc-linear-probe-7245718`). With the fits in parallel on 24 CPUs it took
1 min 53 s. Clay's result was identical in both runs; the spectral baseline
differed by 5 test images (0.8772 vs. 0.8763), most likely a floating-point
effect of the different thread counts.

## 6. Limitations

- EuroSAT is Sentinel-2 Level-1C (top of atmosphere); Clay's normalization
  statistics are for Level-2A. This follows Clay's own EuroSAT code.
- Clay receives zeros for time and location, as in Clay's EuroSAT code,
  although the GeoTIFFs are georeferenced.
- SatMAE sees images resized from 64 to 109 px and center-cropped to 96 px
  (its evaluation transform); Clay sees them at 64 px.
- Frozen features and a linear probe only; one standard split and one run, so
  the intervals reflect only the sampling of test images.
- SatMAE's best C is the largest in the grid, but its validation accuracy had
  levelled off (0.952 at C = 1 and at C = 10).
- The explanations offered for observation 2 and for the spectral difference
  in the run record are hypotheses; the experiment measures the differences,
  not their causes.

## 7. Credits

SatMAE environment versions and the timm 0.3.2 patch follow DevP1130's
SatMAE recipe (PR #28). EuroSAT: Helber et al., 2019. SatMAE: Cong et al.,
NeurIPS 2022. Clay: Clay Foundation, v1.5.
