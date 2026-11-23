#!/usr/bin/env python3
"""Frozen SatMAE features for every EuroSAT image (Python 3.9, torch 1.12, timm 0.3.2).

Model: the pretrained group-channel ViT-Large for multispectral Sentinel-2
(pretrain-vit-large-e199.pth): 96 x 96 input, 8 x 8 patches, three band
groups. Built with models_vit_group_channels.vit_large_patch16 from the
SatMAE repository; the classification head is removed (num_classes=0).

Preprocessing is SatMAE's own EuroSAT evaluation transform
(util/datasets.py: class EuroSat with SatelliteDataset.build_transform):
normalize all 13 bands with SatMAE's EuroSAT mean and std, resize to 109
(bicubic), center-crop to 96, then drop bands 0, 9 and 10.

That code assumes the 13 bands are stored as B01..B08, B8A, B09..B12. Two
variants are written:
  satmae.npz               bands put into that assumed order first (correct input)
  satmae_as_published.npz  file order used as is, exactly as SatMAE's code reads it
When the files already use SatMAE's order, the two are identical and only
the first is written.

Each .npz holds mean [N, 1024] (average of the 3 x 144 patch tokens after
the final LayerNorm, SatMAE's usual global pooling) and cls [N, 1024].
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

EUROSAT_MEAN = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392, 1263.73947144, 1645.40315151,
                1846.87040806, 1762.59530783, 1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
EUROSAT_STD = [633.15169573, 650.2842772, 712.12507725, 965.23119807, 948.9819932, 1108.06650639,
               1258.36394548, 1233.1492281, 1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]
SATMAE_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
DROPPED = [0, 9, 10]
GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
INPUT, PATCH = 96, 8
RESIZE = int(INPUT / (224 / 256))       # 109, as in SatelliteDataset.build_transform
TOKENS = 1 + len(GROUPS) * (INPUT // PATCH) ** 2
ALLOWED_MISSING = {"channel_cls_embed"}  # zero-initialized in the ViT; the pretraining model has no such weight


def build_model(repo, checkpoint, device):
    sys.path.insert(0, str(repo))
    import models_vit_group_channels as mvg
    model = mvg.vit_large_patch16(img_size=INPUT, patch_size=PATCH, in_chans=10, channel_groups=GROUPS,
                                  num_classes=0, drop_path_rate=0.0, global_pool=False)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    missing, unexpected = set(msg.missing_keys), set(msg.unexpected_keys)
    if not missing <= ALLOWED_MISSING:
        raise SystemExit(f"ERROR: encoder weights missing from the checkpoint: {sorted(missing - ALLOWED_MISSING)}")
    if any(not k.startswith(("decoder", "mask_token")) for k in unexpected):
        raise SystemExit(f"ERROR: unexpected non-decoder weights: {sorted(k for k in unexpected if not k.startswith(('decoder', 'mask_token')))}")
    info = {"loaded_tensors": len(state) - len(unexpected), "missing": sorted(missing),
            "ignored_decoder_tensors": len(unexpected), "checkpoint_epoch": ckpt.get("epoch")}
    return model.to(device).eval(), info


def extract(model, pixels, order, device, batch_size):
    tokens = {}
    hook = model.norm.register_forward_hook(lambda m, i, o: tokens.__setitem__("x", o))
    mean = torch.tensor(EUROSAT_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(EUROSAT_STD, device=device)[None, :, None, None]
    keep = [i for i in range(13) if i not in DROPPED]
    n = pixels.shape[0]
    cls = np.empty((n, 1024), np.float32)
    avg = np.empty((n, 1024), np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            x = torch.from_numpy(pixels[s:s + batch_size][:, order].astype(np.float32)).to(device)
            x = (x - mean) / std
            x = TF.resize(x, [RESIZE, RESIZE], interpolation=InterpolationMode.BICUBIC)
            x = TF.center_crop(x, [INPUT, INPUT])[:, keep]
            model.forward_features(x)
            t = tokens["x"]
            if tuple(t.shape) != (x.shape[0], TOKENS, 1024):
                raise SystemExit(f"ERROR: token shape {tuple(t.shape)}, expected {(x.shape[0], TOKENS, 1024)}")
            cls[s:s + x.shape[0]] = t[:, 0].float().cpu().numpy()
            avg[s:s + x.shape[0]] = t[:, 1:].mean(dim=1).float().cpu().numpy()
    hook.remove()
    if not (np.isfinite(cls).all() and np.isfinite(avg).all()):
        raise SystemExit("ERROR: non-finite features")
    return cls, avg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="folder written by prepare_eurosat.py")
    p.add_argument("--repo", required=True, type=Path, help="SatMAE checkout")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="first N images only (testing)")
    a = p.parse_args()

    meta = json.loads((a.data / "meta.json").read_text())
    pixels = np.load(a.data / "pixels.npy", mmap_mode="r")
    if a.limit:
        pixels = pixels[:a.limit]
    file_bands = meta["file_bands"]
    variants = {"satmae": [file_bands.index(b) for b in SATMAE_ORDER]}      # correct input
    if variants["satmae"] != list(range(13)):
        variants["satmae_as_published"] = list(range(13))                    # what SatMAE's code reads

    model, info = build_model(a.repo, a.checkpoint, a.device)
    print("== checkpoint loaded:", json.dumps(info))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, order in variants.items():
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        cls, avg = extract(model, pixels, order, a.device, a.batch_size)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        np.savez(a.out_dir / f"{name}.npz", cls=cls, mean=avg)
        runs[name] = {"band_file_indices_in_satmae_order": order, "images": int(pixels.shape[0]),
                      "seconds": round(time.perf_counter() - start, 1)}
        print(f"== {name}: {runs[name]}", flush=True)
    run = {"model": "SatMAE ViT-Large group-channel, pretrained", "checkpoint": str(a.checkpoint), "load": info,
           "input": {"size": INPUT, "resize": RESIZE, "patch": PATCH, "groups": GROUPS, "dropped": DROPPED},
           "variants": runs, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"}
    (a.out_dir / "satmae.json").write_text(json.dumps(run, indent=2) + "\n")
    print("SATMAE FEATURES COMPLETED")


if __name__ == "__main__":
    main()
