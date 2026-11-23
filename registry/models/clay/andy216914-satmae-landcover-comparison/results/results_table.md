# EuroSAT linear probes (test split, 5400 images)

| Group | Features | Dim | Best C | Test accuracy [95% CI] | Macro-F1 |
|---|---|---|---|---|---|
| main | majority: most frequent training class | - | - | 0.1026 [0.0948, 0.1109] | 0.0186 |
| main | spectral: mean and std of the 10 bands (20 numbers) | 20 | 10.0 | 0.8763 [0.8678, 0.8852] | 0.8710 |
| main | clay: Clay v1.5, class token (1,024) | 1024 | 0.01 | 0.9776 [0.9737, 0.9817] | 0.9765 |
| main | satmae: SatMAE ViT-L, patch average (1,024) | 1024 | 10.0 | 0.9439 [0.9378, 0.9498] | 0.9415 |
| main | clay+satmae: both, concatenated (2,048) | 2048 | 0.01 | 0.9796 [0.9759, 0.9835] | 0.9788 |
| secondary | clay_patch_average: Clay v1.5, patch average | 1024 | 0.1 | 0.9770 [0.9730, 0.9809] | 0.9760 |
| secondary | satmae_class_token: SatMAE ViT-L, class token | 1024 | 1.0 | 0.9457 [0.9398, 0.9519] | 0.9433 |
| band order | satmae_as_published: SatMAE, band order of its published EuroSAT code | 1024 | 1.0 | 0.9507 [0.9450, 0.9563] | 0.9484 |

## Paired differences (exact McNemar test)

| A vs B | A right, B wrong | A wrong, B right | p |
|---|---|---|---|
| clay vs satmae | 241 | 59 | 2.92e-27 |
| clay+satmae vs clay | 26 | 15 | 1.17e-01 |
| clay vs spectral | 591 | 44 | 2.60e-123 |
| satmae vs spectral | 503 | 138 | 1.17e-49 |
| satmae vs satmae_as_published | 122 | 159 | 3.16e-02 |
