#!/usr/bin/env python3
"""Linear probes on frozen features: the same classifier for every input.

For each feature set: standardize (statistics from the training split),
fit multinomial logistic regression for C in {0.01, 0.1, 1, 10}, keep the C
with the best validation accuracy, and evaluate that model once on the test
split. Nothing about the test split is used for any choice. The fits for all
(feature set, C) pairs run in parallel, one process per CPU of the job; each
fit is the same as when run alone.

Feature sets
  main        majority class; spectral stats (mean and std of the 10 bands);
              Clay (class token); SatMAE (patch average); Clay + SatMAE
              (both standardized, then concatenated)
  secondary   the other pooling for each model (Clay patch average, SatMAE class token)
  band order  SatMAE with the band order its published EuroSAT code assumes

Reported: test accuracy with a 95% bootstrap interval (2,000 resamples of
the test set), macro-F1, per-class accuracy, confusion matrices, and exact
McNemar tests for paired differences on the same test images.

Writes <results>/metrics.json, results_table.md, test_predictions.csv and
figures/{confusion_matrices,per_class_accuracy}.png.
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0]
SEED = 0


def standardize(x, train):
    return StandardScaler().fit(x[train]).transform(x).astype(np.float32)


def feature_sets(data, emb):
    """name -> (group, description, matrix or None for the majority baseline)."""
    stats = np.load(data / "spectral_stats.npy")
    clay, satmae = np.load(emb / "clay.npz"), np.load(emb / "satmae.npz")
    sets = {
        "majority": ("main", "most frequent training class", None),
        "spectral": ("main", "mean and std of the 10 bands (20 numbers)", stats),
        "clay": ("main", "Clay v1.5, class token (1,024)", clay["cls"]),
        "satmae": ("main", "SatMAE ViT-L, patch average (1,024)", satmae["mean"]),
        "clay+satmae": ("main", "both, concatenated (2,048)", "concat"),
        "clay_patch_average": ("secondary", "Clay v1.5, patch average", clay["mean"]),
        "satmae_class_token": ("secondary", "SatMAE ViT-L, class token", satmae["cls"]),
    }
    published = emb / "satmae_as_published.npz"
    if published.exists():
        sets["satmae_as_published"] = ("band order", "SatMAE, band order of its published EuroSAT code",
                                       np.load(published)["mean"])
    return sets


def fit(x, y, tr, va, te, c):
    """One probe: returns validation accuracy, test predictions, whether it failed to converge, seconds."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m = LogisticRegression(C=c, max_iter=5000, random_state=SEED).fit(x[tr], y[tr])
    not_converged = any(issubclass(x_.category, ConvergenceWarning) for x_ in w)
    return float((m.predict(x[va]) == y[va]).mean()), m.predict(x[te]), not_converged, time.perf_counter() - t0


def bootstrap_ci(correct, n_boot=2000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    acc = correct[idx].mean(axis=1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def mcnemar(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum())     # A right, B wrong
    c = int((~correct_a & correct_b).sum())     # A wrong, B right
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="folder written by prepare_eurosat.py")
    p.add_argument("--emb", required=True, type=Path, help="folder with clay.npz and satmae*.npz")
    p.add_argument("--results", required=True, type=Path)
    a = p.parse_args()

    meta = json.loads((a.data / "meta.json").read_text())
    classes = meta["classes"]
    y = np.load(a.data / "labels.npy")
    split = np.load(a.data / "split.npy")
    tr, va, te = split == 0, split == 1, split == 2
    sets = feature_sets(a.data, a.emb)
    mats = {}
    for name, (group, desc, x) in sets.items():
        if isinstance(x, str):              # concatenation of the two standardized main features
            mats[name] = np.hstack([mats["clay"], mats["satmae"]])
        elif x is not None:
            mats[name] = standardize(x, tr)
    tasks = sorted(((k, c) for k in mats for c in C_GRID), key=lambda t: -mats[t[0]].shape[1] * t[1])  # slowest first
    workers = min(len(tasks), int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1))
    print(f"== {len(tasks)} fits on {workers} parallel workers", flush=True)
    t0 = time.perf_counter()
    fits = dict(zip(tasks, Parallel(n_jobs=workers)(delayed(fit)(mats[k], y, tr, va, te, c) for k, c in tasks)))
    print(f"== all fits done in {time.perf_counter() - t0:.0f} s", flush=True)
    results, preds = {}, {}
    for name, (group, desc, x) in sets.items():
        if x is None:
            majority = np.bincount(y[tr]).argmax()
            pred, best_c, val_acc, seconds = np.full(te.sum(), majority), None, {}, 0.0
        else:
            x = mats[name]
            val_acc = {str(c): fits[(name, c)][0] for c in C_GRID}
            best_c = max(C_GRID, key=lambda c: (val_acc[str(c)], -c))
            _, pred, not_converged, _ = fits[(name, best_c)]
            seconds = sum(fits[(name, c)][3] for c in C_GRID)
        correct = pred == y[te]
        lo, hi = bootstrap_ci(correct)
        cm = confusion_matrix(y[te], pred, labels=range(len(classes)))
        results[name] = {
            "group": group, "description": desc, "features": None if x is None else int(x.shape[1]),
            "best_C": best_c, "validation_accuracy_by_C": val_acc,
            "test_accuracy": float(correct.mean()), "test_accuracy_ci95": [lo, hi],
            "test_macro_f1": float(f1_score(y[te], pred, average="macro")),
            "per_class_accuracy": {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist(),
            "converged": None if x is None else not not_converged,
            "fit_seconds_all_C": round(seconds, 1),
        }
        preds[name] = pred
        print(f"== {name:22s} test accuracy {correct.mean():.4f} [{lo:.4f}, {hi:.4f}]  best C {best_c}", flush=True)

    ok = {k: preds[k] == y[te] for k in preds}
    best_single = max(("clay", "satmae"), key=lambda k: results[k]["test_accuracy"])
    tests = {"clay vs satmae": mcnemar(ok["clay"], ok["satmae"]),
             f"clay+satmae vs {best_single}": mcnemar(ok["clay+satmae"], ok[best_single]),
             "clay vs spectral": mcnemar(ok["clay"], ok["spectral"]),
             "satmae vs spectral": mcnemar(ok["satmae"], ok["spectral"])}
    if "satmae_as_published" in ok:
        tests["satmae vs satmae_as_published"] = mcnemar(ok["satmae"], ok["satmae_as_published"])

    a.results.mkdir(parents=True, exist_ok=True)
    out = {"split_sizes": meta["split_sizes"], "classes": classes, "C_grid": C_GRID,
           "classifier": "sklearn LogisticRegression (lbfgs, multinomial), max_iter 5000, standardized features",
           "parallel_workers": workers,
           "results": results, "paired_tests_mcnemar_exact": tests}
    (a.results / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    header = "| Group | Features | Dim | Best C | Test accuracy [95% CI] | Macro-F1 |\n|---|---|---|---|---|---|\n"
    rows = "".join(f"| {r['group']} | {k}: {r['description']} | {r['features'] or '-'} | {r['best_C'] or '-'} | "
                   f"{r['test_accuracy']:.4f} [{r['test_accuracy_ci95'][0]:.4f}, {r['test_accuracy_ci95'][1]:.4f}] | "
                   f"{r['test_macro_f1']:.4f} |\n" for k, r in results.items())
    trows = "".join(f"| {k} | {v['a_right_b_wrong']} | {v['a_wrong_b_right']} | {v['p_value']:.2e} |\n" for k, v in tests.items())
    (a.results / "results_table.md").write_text(
        f"# EuroSAT linear probes (test split, {int(te.sum())} images)\n\n{header}{rows}\n"
        f"## Paired differences (exact McNemar test)\n\n| A vs B | A right, B wrong | A wrong, B right | p |\n"
        f"|---|---|---|---|\n{trows}")
    with open(a.results / "test_predictions.csv", "w") as f:
        f.write("index,label," + ",".join(preds) + "\n")
        for i, j in enumerate(np.flatnonzero(te)):
            f.write(f"{j},{classes[y[j]]}," + ",".join(classes[preds[k][i]] for k in preds) + "\n")

    fig_dir = a.results / "figures"
    fig_dir.mkdir(exist_ok=True)
    short = [c.replace("HerbaceousVegetation", "Herbaceous").replace("PermanentCrop", "Perm.Crop") for c in classes]
    shown = ["clay", "satmae", "clay+satmae"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, k in zip(axes, shown):
        cm = np.array(results[k]["confusion_matrix"], float)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"{k}  (accuracy {results[k]['test_accuracy']:.3f})", fontsize=10)
        ax.set_xticks(range(len(classes)), short, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)), short, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrices.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.27
    for i, (k, shade) in enumerate(zip(shown, ["0.75", "0.45", "0.1"])):
        vals = [results[k]["per_class_accuracy"][c] for c in classes]
        ax.bar(np.arange(len(classes)) + (i - 1) * width, vals, width, color=shade, label=k)
    ax.set_xticks(range(len(classes)), short, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "per_class_accuracy.png", dpi=160); plt.close(fig)
    print("PROBES COMPLETED")


if __name__ == "__main__":
    main()
