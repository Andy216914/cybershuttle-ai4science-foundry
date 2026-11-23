#!/usr/bin/env python3
"""Frozen Clay v1.5 features for every EuroSAT image.

Settings follow Clay's own EuroSAT code at commit f14e698
(claymodel/finetune/classify/eurosat_datamodule.py and eurosat_model.py):
10 Sentinel-2 bands, normalized with the sentinel-2-l2a mean and std from
Clay's configs/metadata.yaml, those bands' wavelengths, 10 m ground sample
distance, and zeros for time and location (Clay's code fills them with
zeros for EuroSAT). Images stay at 64 x 64, i.e. 8 x 8 patches.

The model is loaded with load_encoder() from the Clay inference recipe next
to this folder, the loader whose output matched Clay's published embeddings.

Writes <out> (.npz): cls [N, 1024] (class token, the feature Clay's own
classifier uses) and mean [N, 1024] (average of the 64 patch tokens).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "andy216914-embedding-inference" / "scripts"))
from clay_embed import load_encoder  # noqa: E402

CLAY_EUROSAT_WAVES = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]   # eurosat_model.py
CLAY_BAND_NAMES = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
EMBED_DIM, TOKENS = 1024, 1 + 8 * 8


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="folder written by prepare_eurosat.py")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path, help="Clay's configs/metadata.yaml")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    meta = json.loads((a.data / "meta.json").read_text())
    idx = meta["model_band_file_indices"]                      # B02 ... B12 in Clay's band order
    s2 = yaml.safe_load(a.metadata.read_text())["sentinel-2-l2a"]
    if s2["band_order"] != CLAY_BAND_NAMES:
        raise SystemExit(f"ERROR: unexpected Clay band order {s2['band_order']}")
    mean = np.array([s2["bands"]["mean"][b] for b in CLAY_BAND_NAMES], np.float32)[None, :, None, None]
    std = np.array([s2["bands"]["std"][b] for b in CLAY_BAND_NAMES], np.float32)[None, :, None, None]
    waves = [float(s2["bands"]["wavelength"][b]) for b in CLAY_BAND_NAMES]
    if waves != CLAY_EUROSAT_WAVES:
        raise SystemExit(f"ERROR: wavelengths {waves} differ from Clay's EuroSAT code {CLAY_EUROSAT_WAVES}")

    encoder, info = load_encoder(a.checkpoint, a.metadata, a.device)
    pixels = np.load(a.data / "pixels.npy", mmap_mode="r")
    n = pixels.shape[0]
    cls = np.empty((n, EMBED_DIM), np.float32)
    avg = np.empty((n, EMBED_DIM), np.float32)
    gsd = torch.tensor(10.0, device=a.device)
    wv = torch.tensor(waves, device=a.device)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n, a.batch_size):
            x = (pixels[s:s + a.batch_size][:, idx].astype(np.float32) - mean) / std
            b = x.shape[0]
            cube = {"pixels": torch.from_numpy(x).to(a.device),
                    "time": torch.zeros(b, 4, device=a.device),
                    "latlon": torch.zeros(b, 4, device=a.device),
                    "gsd": gsd, "waves": wv}
            enc, *_ = encoder(cube)
            if tuple(enc.shape) != (b, TOKENS, EMBED_DIM):
                raise SystemExit(f"ERROR: encoder output {tuple(enc.shape)}, expected {(b, TOKENS, EMBED_DIM)}")
            cls[s:s + b] = enc[:, 0].float().cpu().numpy()
            avg[s:s + b] = enc[:, 1:].mean(dim=1).float().cpu().numpy()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if not (np.isfinite(cls).all() and np.isfinite(avg).all()):
        raise SystemExit("ERROR: non-finite features")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, cls=cls, mean=avg)
    run = {"model": "Clay v1.5", "images": n, "seconds": round(seconds, 1),
           "gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "bands_file_indices": idx, "band_layout": meta["layout"], "gsd": 10.0, "waves": waves,
           "time_latlon": "zeros, as in Clay's EuroSAT code", "loader": info}
    a.out.with_suffix(".json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print("CLAY FEATURES COMPLETED")


if __name__ == "__main__":
    main()
