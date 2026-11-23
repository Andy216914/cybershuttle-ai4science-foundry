#!/usr/bin/env python3
"""Pack EuroSAT into one array file and determine its band order from the data.

Both models read the packed file, so they see exactly the same pixels.

Band order. The EuroSAT GeoTIFFs carry 13 bands without names, and two
reference implementations disagree on their order:
  torchgeo  B01..B08, B09, B10, B11, B12, B8A   (TorchGeo, used by Clay's EuroSAT code)
  satmae    B01..B08, B8A, B09, B10, B11, B12   (SatMAE's EuroSat class, util/datasets.py)
The cirrus band B10 is nearly zero in top-of-atmosphere imagery, so its
position identifies the layout: file index 9 means torchgeo, 10 means satmae.
The correlation of each band with B08 (file index 7) is reported as a second
check: B8A, the narrow near-infrared band, tracks B08 almost exactly.

Writes:
  <out>/pixels.npy            uint16 [N, 13, 64, 64], file band order
  <out>/labels.npy, split.npy int64 class index, int8 0=train 1=val 2=test
  <out>/spectral_stats.npy    float32 [N, 20]: per-image mean and std of the 10 bands both models use
  <out>/meta.json             classes, split sizes, band layout, band indices
  <results>/band_order.json   per-band statistics and the decision
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]   # alphabetical, as in TorchGeo
LAYOUTS = {
    "torchgeo": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"],
    "satmae":   ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
}
CIRRUS_INDEX = {"torchgeo": 9, "satmae": 10}
MODEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]   # both models: 10 bands, no B01/B09/B10
SATMAE_PUBLISHED_MEAN = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392, 1263.73947144, 1645.40315151,
                         1846.87040806, 1762.59530783, 1972.62420416, 582.72633433, 14.77112979, 1732.16362238,
                         1247.91870117]   # SatMAE util/datasets.py, class EuroSat


def read_split(path):
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return [n.replace(".jpg", ".tif") for n in names]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tif-dir", required=True, type=Path, help="folder with one sub-folder of .tif files per class")
    p.add_argument("--split-dir", required=True, type=Path, help="folder with eurosat-{train,val,test}.txt")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    a = p.parse_args()

    splits = {s: read_split(a.split_dir / f"eurosat-{s}.txt") for s in ("train", "val", "test")}
    names = splits["train"] + splits["val"] + splits["test"]
    split = np.concatenate([np.full(len(splits[s]), i, np.int8) for i, s in enumerate(("train", "val", "test"))])
    if len(set(names)) != len(names):
        raise SystemExit("ERROR: an image appears in more than one split")
    on_disk = sorted(f"{c}/{f.name}" for c in CLASSES for f in (a.tif_dir / c).glob("*.tif"))
    if sorted(f"{n.rsplit('_', 1)[0]}/{n}" for n in names) != on_disk:
        raise SystemExit(f"ERROR: the split files list {len(names)} images, the folder has {len(on_disk)}")
    labels = np.array([CLASSES.index(n.rsplit("_", 1)[0]) for n in names], np.int64)

    a.out.mkdir(parents=True, exist_ok=True)
    n = len(names)
    pixels = np.lib.format.open_memmap(a.out / "pixels.npy", mode="w+", dtype=np.uint16, shape=(n, 13, 64, 64))
    band_mean = np.zeros((n, 13), np.float64)          # per-image band means, for the audit
    for i, name in enumerate(names):
        with rasterio.open(a.tif_dir / name.rsplit("_", 1)[0] / name) as src:
            img = src.read()
        if img.shape != (13, 64, 64) or img.dtype != np.uint16:
            raise SystemExit(f"ERROR: {name} is {img.shape} {img.dtype}, expected (13, 64, 64) uint16")
        pixels[i] = img
        band_mean[i] = img.reshape(13, -1).mean(axis=1)
        if (i + 1) % 5000 == 0:
            print(f"   packed {i + 1} / {n}", flush=True)
    pixels.flush()

    # ---- band-order audit ----
    mean = band_mean.mean(axis=0)
    corr_b08 = np.corrcoef(band_mean.T)[7]
    darkest = int(mean.argmin())
    layout = next((k for k, v in CIRRUS_INDEX.items() if v == darkest), None)
    if layout is None:
        raise SystemExit(f"ERROR: darkest band at file index {darkest}; matches neither known layout")
    names13 = LAYOUTS[layout]
    b8a = names13.index("B8A")
    audit = {
        "decision": layout,
        "rule": "file index of the darkest band (cirrus B10): 9 = torchgeo layout, 10 = satmae layout",
        "darkest_band_file_index": darkest,
        "b8a_file_index": b8a,
        "b8a_correlation_with_b08": round(float(corr_b08[b8a]), 4),
        "per_file_index": [{"file_index": i, "band": names13[i], "mean_dn": round(float(mean[i]), 1),
                            "corr_with_file_index_7_b08": round(float(corr_b08[i]), 4),
                            "satmae_published_mean": round(SATMAE_PUBLISHED_MEAN[i], 1)} for i in range(13)],
        "consequence": {
            "clay": "Clay's EuroSAT code selects bands by TorchGeo's names: "
                    + ("correct for this copy." if layout == "torchgeo" else "wrong for this copy."),
            "satmae": "SatMAE's EuroSat class assumes the satmae layout (drops file indices 0, 9, 10): "
                      + ("correct for this copy." if layout == "satmae" else
                         "wrong for this copy; it would feed B09 as B8A and shift the shortwave-infrared bands."),
        },
        "n_images": n,
    }
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / "band_order.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"== band layout: {layout} (darkest band at file index {darkest}; "
          f"B8A at {b8a}, correlation with B08 {corr_b08[b8a]:.3f})")

    # ---- simple baseline features: mean and std of each of the 10 bands ----
    idx = [names13.index(b) for b in MODEL_BANDS]
    stats = np.empty((n, 20), np.float32)
    for s in range(0, n, 2000):
        x = pixels[s:s + 2000][:, idx].astype(np.float32).reshape(-1, 10, 64 * 64)
        stats[s:s + 2000, :10] = x.mean(axis=2)
        stats[s:s + 2000, 10:] = x.std(axis=2)
    np.save(a.out / "labels.npy", labels)
    np.save(a.out / "split.npy", split)
    np.save(a.out / "spectral_stats.npy", stats)
    meta = {
        "classes": CLASSES,
        "n": n,
        "split_sizes": {s: int((split == i).sum()) for i, s in enumerate(("train", "val", "test"))},
        "layout": layout,
        "file_bands": names13,
        "model_bands": MODEL_BANDS,
        "model_band_file_indices": idx,
        "names_file": "names.txt",
    }
    (a.out / "names.txt").write_text("\n".join(names) + "\n")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"== packed {n} images: {meta['split_sizes']}")
    print("PREPARE COMPLETED")


if __name__ == "__main__":
    main()
