#!/usr/bin/env python3
"""Score one project's HiDRA predictions against hand labels kept in another project.

Outputs:
  vs_handlabels_snapshot.png   ROC and precision-recall of the current predictions
  vs_handlabels_over_time.png  past review decisions rescored against the hand labels. Each one
                               logged the model's confidence when it was reviewed, so this needs no
                               old checkpoints.

By default only frames with an explicit label are scored. Pass --full-coverage if every real bout
in the social window was painted. Then every other frame in the window counts as a negative.

    python lab_scripts/compare_to_handlabels.py \\
        --project ~/my-projects/approach_defs --video 0 \\
        --gt-project ~/my-projects/cage1rb_day1 --gt-video 0 \\
        --behavior 0 --track 1 --full-coverage

Needs pandas/numpy/sklearn/matplotlib (the labeler's own venv has all four).
"""
import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def load_events(proj: Path):
    ev = []
    for f in sorted(glob.glob(str(proj / "events" / "*.jsonl"))):
        for line in open(f):
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="project whose HiDRA predictions you're scoring")
    ap.add_argument("--video", required=True, help="video_id in --project")
    ap.add_argument("--gt-project", required=True, help="project holding the hand labels (ground truth)")
    ap.add_argument("--gt-video", default=None, help="video_id in --gt-project (default: same as --video)")
    ap.add_argument("--behavior", type=int, required=True)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--full-coverage", action="store_true",
                    help="count every unpainted frame in the social window as a negative. Only use "
                         "this if every real bout was painted.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proj = Path(a.project).expanduser()
    gt_proj = Path(a.gt_project).expanduser()
    gt_vid = a.gt_video or a.video
    out = Path(a.out) if a.out else proj / "analysis" / "vs_handlabels"
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(proj / "project.json"))
    beh = next(b for b in manifest["behaviors"] if b["id"] == a.behavior)
    thr = (beh.get("postproc") or {}).get("hi", 0.6)
    n_frames = next(v["n_frames"] for v in manifest["videos"] if v["video_id"] == a.video)

    # Ground truth from the hand-labeled project.
    gt_all = pd.read_parquet(gt_proj / "labels" / f"{gt_vid}.parquet")
    gt = gt_all[(gt_all.track == a.track) & (gt_all.behavior_id == a.behavior) & gt_all.value.isin([0, 1])]
    if gt.empty:
        raise SystemExit(f"no hand labels found in {gt_proj}/labels/{gt_vid}.parquet for "
                         f"behavior {a.behavior}, track {a.track}")

    if a.full_coverage:
        gt_manifest = json.load(open(gt_proj / "project.json"))
        gt_entry = next(v for v in gt_manifest["videos"] if v["video_id"] == gt_vid)
        ws = gt_entry.get("social_start"); we = gt_entry.get("social_end")
        if ws is None or we is None:
            raise SystemExit(f"--full-coverage needs a social window on {gt_vid} in {gt_proj}. "
                             "It marks the part of the video you reviewed.")
        y_lookup = np.full(n_frames, -1, dtype=int)
        y_lookup[ws:we] = 0
        pos = gt[gt.value == 1].frame.to_numpy()
        y_lookup[pos[(pos >= ws) & (pos < we)]] = 1
        print(f"--full-coverage: scoring all {we - ws} frames in the social window [{ws},{we}). "
             f"{int((y_lookup[ws:we]==1).sum())} are positive. The rest count as negative.")
    else:
        y_lookup = np.full(n_frames, -1, dtype=int)
        y_lookup[gt.frame.to_numpy()] = gt.value.to_numpy()
        span = int(gt.frame.max() - gt.frame.min())
        coverage = len(gt) / span * 100 if span else 0.0
        print(f"Scoring only the {len(gt)} labeled frames ({coverage:.1f}% of the labeled span). "
             "Pass --full-coverage if you reviewed the whole window.")

    # Current predictions vs. hand labels.
    proba_full = np.load(proj / "predictions" / a.video / f"{a.behavior}.npy")[:, a.track]
    frames = np.flatnonzero(y_lookup >= 0)
    y_true = y_lookup[frames].astype(int)
    y_score = proba_full[frames]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_score)
    ap_score = average_precision_score(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    base_rate = (y_true == 1).mean()

    print(f"{a.project}/{a.video} predictions vs. {a.gt_project}/{gt_vid} hand labels "
         f"(behavior {a.behavior}, track {a.track})")
    print(f"scored frames: {len(frames)} ({int((y_true==1).sum())} pos, {int((y_true==0).sum())} neg), "
         f"positive rate {base_rate:.1%}")
    print(f"at live threshold >= {thr:.2f}: accuracy {accuracy:.3f}  precision {precision:.3f}  "
         f"recall {recall:.3f}  specificity {specificity:.3f}  f1 {f1:.3f}")
    print(f"AUROC {auroc:.3f}   average precision {ap_score:.3f}  (chance = {base_rate:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    ax.plot(fpr, tpr, color=CAT[0], lw=2.4, solid_capstyle="round")
    ax.fill_between(fpr, tpr, color=CAT[0], alpha=0.08, lw=0)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.grid(True, lw=0.6, alpha=0.6)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC vs. hand labels\nAUROC {auroc:.2f}")

    ax = axes[1]
    ax.plot(pr_r, pr_p, color=CAT[0], lw=2.4, solid_capstyle="round")
    ax.axhline(base_rate, color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=0)
    ax.plot(recall, precision, marker="o", ms=7, color=INK, mec=SURFACE, mew=1, zorder=5,
           label=f"operating pt. (thr={thr:.2f})")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.grid(True, lw=0.6, alpha=0.6)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall vs. hand labels\nAP {ap_score:.2f} (chance = {base_rate:.2f})")
    ax.legend(loc="upper right")

    fig.suptitle(f"{a.video} in {proj.name}: HiDRA predictions vs. hand labels in {gt_proj.name}",
                fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "vs_handlabels_snapshot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame([dict(video=a.video, gt_video=gt_vid, full_coverage=a.full_coverage,
                       n_scored_frames=len(frames), n_pos=int((y_true == 1).sum()),
                       n_neg=int((y_true == 0).sum()), base_rate=base_rate,
                       threshold=thr, accuracy=accuracy, precision=precision, recall=recall,
                       specificity=specificity, f1=f1, auroc=auroc, ap=ap_score,
                       tp=tp, fp=fp, tn=tn, fn=fn)]).to_csv(out / "vs_handlabels_snapshot.csv", index=False)

    # Past review decisions on this clip, rescored against the hand labels.
    ev = load_events(proj)
    dec = [e for e in ev if e.get("type") in ("candidate_accept", "candidate_reject")
          and e.get("behavior_id") == a.behavior and e.get("track") == a.track
          and e.get("video_id") == a.video and e.get("t_srv") and e.get("proba") is not None]
    trains = sorted(e["t_srv"] for e in ev if e.get("type") == "job_done" and e.get("kind") == "train"
                    and e.get("status") == "done")
    if not dec:
        print(f"\nNo past review decisions for {a.video} in {proj.name}. Skipping the over-time plot.")
        return

    rows = []
    for e in dec:
        seg = y_lookup[e["start"]:e["end"]]
        seg = seg[seg >= 0]
        if len(seg) < 5:
            continue
        true_frac = float((seg == 1).mean())
        rows.append(dict(t=e["t_srv"], proba=float(e["proba"]), true_label=int(true_frac >= 0.5),
                         true_frac=true_frac, n_gt_frames=len(seg)))
    if not rows:
        print(f"\nNone of the {len(dec)} past review decisions for {a.video} overlap a hand-labeled "
             "frame. Skipping the over-time plot.")
        return
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").reset_index(drop=True)
    df["correct_at_thr"] = ((df["proba"] >= thr).astype(int) == df["true_label"]).astype(int)
    win = max(8, min(25, len(df) // 5 or 8))
    df["rolling_acc"] = df["correct_at_thr"].rolling(win, min_periods=max(4, win // 3)).mean()

    overall_auroc = (roc_auc_score(df["true_label"], df["proba"])
                    if df["true_label"].nunique() == 2 else float("nan"))
    overall_acc = df["correct_at_thr"].mean()
    print(f"\nover time ({len(df)} historical review decisions on {a.video} with hand-label overlap):")
    print(f"  overall accuracy-at-decision-time vs. hand labels: {overall_acc:.3f}   "
         f"AUROC (proba vs. hand label): {overall_auroc:.3f}")

    fig, ax = plt.subplots(figsize=(9.8, 5))
    jitter = np.random.uniform(-0.03, 0.03, len(df))
    right = (df["true_label"] == 1).to_numpy()
    ax.scatter(df["t"][right], (df["proba"][right] >= thr).astype(int) + jitter[right],
              s=16, color=CAT[2], alpha=0.65, label="hand labels say: real approach", zorder=3)
    ax.scatter(df["t"][~right], (df["proba"][~right] >= thr).astype(int) + jitter[~right],
              s=16, color=CAT[1], alpha=0.65, label="hand labels say: not an approach", zorder=3)
    ax.plot(df["t"], df["rolling_acc"], color=INK, lw=2.2, zorder=5,
           label=f"rolling accuracy vs. hand labels (window={win})")
    for i, t in enumerate(trains):
        ax.axvline(pd.to_datetime(t), color=GOOD, lw=1.2, ls=(0, (3, 2)), alpha=0.8,
                  label="fine-tune completed" if i == 0 else None, zorder=1)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 1], ["called 'not approach'\n(proba < thr)", "called 'approach'\n(proba ≥ thr)"])
    ax.set_xlabel("review decision, in time order (colored by what the hand labels actually say)")
    ax.grid(True, axis="y", lw=0.6, alpha=0.6)
    ax.set_title(f"{a.video}: was the model's call right, checked against hand labels, over time",
                fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out / "vs_handlabels_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    df.drop(columns=["t"]).assign(t=df["t"].astype(str)).to_csv(
        out / "vs_handlabels_over_time.csv", index=False)
    print(f"\nwrote {out}/vs_handlabels_snapshot.{{png,csv}} and vs_handlabels_over_time.{{png,csv}}")


if __name__ == "__main__":
    main()
