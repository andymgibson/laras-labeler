# lab_scripts — Kennedy lab tools for the sDANNCE / rat social-behavior data

This repo is a copy of [talmolab/laras-labeler](https://github.com/talmolab/laras-labeler) (Lara Scaria's
human-in-the-loop labeler, BSD-3-Clause) plus the lab's own changes. Everything under `lab_scripts/`
is ours; upstream never touches this folder, so pulling Lara's updates cannot conflict with it.

## Scripts

| script | what it does |
|---|---|
| `dannce_to_slp.py` | Converts one camera view of an sDANNCE `save_data_AVG0.mat` (3D poses, mm) into a per-camera 2D `.slp` the labeler can load next to that camera's video, using the cage's `calibration.json`. |
| `import_social_window.py` | Sets a clip's *social window* (frames where the second rat is in the arena) from the QC spreadsheet, through a running labeler's HTTP API. |

## Known issue: joints come out as `kp00 … kp22`

`dannce_to_slp.py` writes anonymous node names. The labeler needs real names to compute heading/facing/
approach features (with anonymous names it silently drops 20 features), and HiDRA needs them too.
The rat23 order (from the s-DANNCE paper's `sDANNCE_main.m`, checked against our data) is:

`1 Snout, 2 EarL, 3 EarR, 4 SpineF, 5 SpineM, 6 SpineL, 7 TailBase, 8-11 left forelimb
(shoulder, elbow, wrist, hand), 12-15 right forelimb, 16-19 left hindlimb (hip, knee, ankle, foot),
20-23 right hindlimb`

Until the script is updated, rename the nodes after conversion (nose, ear_left, ear_right, neck,
body_center, spine_lower, tail_base, shoulder_left, ... — the names the labeler's feature code looks for).

## What we changed in the labeler itself

- **Social window** (per clip `social_start`/`social_end`): features, training, review candidates and label
  writes are restricted to the frames where a second animal is actually present; the GUI dims the rest.
- Adding a video with no poses (labeling-only mode) no longer fails.

## Pulling Lara's updates

`upstream` points at talmolab/laras-labeler (push is disabled on purpose).

```bash
git fetch upstream
git checkout -b sync-upstream
git merge upstream/main          # resolve conflicts if any, then run the labeler and try it
git push -u origin sync-upstream # open a pull request into main
```

## Data

Videos, poses and projects are **not** in this repo. They live on the lab server
(`smb://labdata.scripps.edu/kennedy/...`) and in each user's `~/my-projects`.
