# lab_scripts — Kennedy lab tools for the sDANNCE / rat social-behavior data

This repo is a copy of [talmolab/laras-labeler](https://github.com/talmolab/laras-labeler) (Lara Scaria's
human-in-the-loop labeler, BSD-3-Clause) plus the lab's own changes. Everything under `lab_scripts/`
is ours; upstream never touches this folder, so pulling Lara's updates cannot conflict with it.

## Scripts

| script | what it does |
|---|---|
| `dannce_to_slp.py` | Converts one camera view of an sDANNCE `save_data_AVG0.mat` (3D poses, mm) into a per-camera 2D `.slp` the labeler can load next to that camera's video, using the cage's `calibration.json`. |
| `import_social_window.py` | Sets a clip's *social window* (frames where the second rat is in the arena) from the QC spreadsheet, through a running labeler's HTTP API. |
| `review_stats.py` | Precision of a head's proposals from your accept and reject decisions in review. |
| `characterize_approach_head.py` | ROC, precision-recall and review precision over time for every video in a project, scored against that project's own labels. |
| `compare_to_handlabels.py` | Scores one video's predictions against hand labels kept in a different project. Add `--full-coverage` when every real bout was painted. |

## The pixel scale (px/cm) for HiDRA is measured automatically

HiDRA works in centimetres, so it needs each camera's px/cm. `dannce_to_slp.py` measures it while
converting (true 3D size in mm vs. projected size in pixels, over long head/trunk segments of the whole
session) and writes `<poses>.slp.scale.json` next to the `.slp`. **Keep the two files together** when you
copy poses between machines. The labeler then fills the px/cm box by itself:

- a value you type always wins; clear the box to go back to the automatic one;
- a video that is a uniformly resized copy gets the scale multiplied by the same factor;
- a video of a different shape (e.g. cropped) is refused with the reason shown, never guessed;
- no sidecar → nothing is guessed, and HiDRA still asks for the number.

To measure it for poses you already converted: `python dannce_to_slp.py <cage_dir> --camera CameraN --out
<poses>.slp --scale-only`. It is one number per camera, so it is approximate: the sidecar records the
spread (`iqr`), which is wide because a tilted camera sees the arena at different scales near and far.

## What we changed in the labeler itself

- **Social window** (per clip `social_start`/`social_end`): features, training, review candidates and label
  writes are restricted to the frames where a second animal is actually present; the GUI dims the rest.
- Adding a video with no poses (labeling-only mode) no longer fails.
- `HIDRA_GPU=<n>` picks the GPU for HiDRA jobs. Set it when you start the labeler.
- `?ft_mode=head` (or `tail`, `embedding`) on the train route picks the fine-tune mode. Later re-tunes
  keep using the mode of the last successful run.

## Bugs we found and fixed

**Changing the review rate undid the fine-tune.** Fixed 2026-10-20.
Editing the "review rate" box or the "scores" menu saved a new copy of the head's settings. That copy
left out the link to the fine-tuned model. Predict then went back to the stock model without saying
so. The next re-tune also ran in the slow tail mode. Now the link is kept as long as you stay on the
same head. Picking a different head still starts fresh, because the old fine-tune belongs to the old head.

**Fine-tuning on several videos only used one of them.** Fixed 2026-10-09.
Every camera video in this dataset is named `0.mp4`. Fine-tuning named each video's exported poses
after that file name, so each video overwrote the one before it. A fine-tune that seemed to use five
videos really used only the last one. Exports are now named by the clip's ID instead. Redo any
analysis that came from a fine-tune before this date. To check a fine-tune, open
`<project>/hidra/_finetune/<lab>__<action>/tracking/`. It should hold one file per video.

**HiDRA ignored the GPU we picked.** Fixed 2026-10-06.
HiDRA always ran on GPU 0, even when the labeler was started with `CUDA_VISIBLE_DEVICES`. On a shared
machine that GPU is often full and the job crashed out of memory. Use `HIDRA_GPU` instead (see above).

**Joint names came out as `kp00` to `kp22`.** Fixed 2026-10-06.
`dannce_to_slp.py` now writes the real names (nose, ear_left and so on). Without them the labeler
dropped about 20 features unless you renamed the nodes by hand.

**The trim handle sat one frame past the playhead.** Fixed 2026-10-05.
When you dragged the end of a candidate, its handle was drawn one frame to the right of the playhead.
Only the drawing was off. The saved labels were always right.

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
