# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "openpyxl"]
# ///
"""Set a laras-labeler clip's social window (the frame range the second rat is actually in the
arena) from Hannah's QC spreadsheet, via the running labeler's HTTP API.

`asd_session_qc_social.xlsx` has two sheets: the main `asd_session_qc_social` sheet (day, cage,
animal_id, entry_time/exit_time in MINUTES, ...) and `Sheet1`, which has the same entry_time/
exit_time plus precomputed `start_frame`/`end_frame`. The two sheets share no explicit key column,
but are row-aligned -- verified by cross-checking entry_time/exit_time (duplicated in both) match
row-for-row. We use `start_frame`/`end_frame` from Sheet1 AS GIVEN rather than recomputing frame
numbers from entry_time/exit_time x fps: cross-checked against Camera1/cage1RB/day1's own
frametimes.npy and a constant-fps conversion doesn't land on the same frame numbers (capture isn't
perfectly constant-rate), so the sheet's own precomputed numbers are the more trustworthy source.

Video ids in this project are just the source filename stem (e.g. "0" for .../Camera1/0.mp4), not
descriptive, so this script does not try to auto-match a project's clips to a spreadsheet row --
you look up the row with --list, then apply it to a specific --pid/--vid.

This has nothing to import for alone-condition sessions (single rat, no entry/exit event) -- they
are not in this spreadsheet at all. Just never call this for those clips; a clip with no window set
behaves exactly as before (whole video usable).

Usage:
    # see every row (day, cage, animal_id, entry/exit minutes, start/end frame)
    uv run scripts/import_social_window.py --list

    # find cage1RB on day1 and print its row
    uv run scripts/import_social_window.py --day day1 --cage cage1RB

    # apply it to a clip in a running labeler project
    uv run scripts/import_social_window.py --day day1 --cage cage1RB --pid cage1rb_day1 --vid 0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

DEFAULT_XLSX = ("/Volumes/kennedy/DATA/GLOBUS_SYNC/Hannah_data/ASD_Social_Photometry/"
                "asd_session_qc_social.xlsx")


def load_sessions(xlsx_path: str) -> pd.DataFrame:
    main = pd.read_excel(xlsx_path, sheet_name="asd_session_qc_social")
    frames = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    if len(main) != len(frames):
        sys.exit(f"sheet row-count mismatch ({len(main)} vs {len(frames)}) -- "
                 f"row-position join is no longer safe, this script needs updating")
    mismatched = (main["entry_time"].round(2) != frames["entry_time"].round(2)).sum()
    if mismatched:
        sys.exit(f"{mismatched} row(s) where the two sheets' entry_time disagree -- "
                 f"row-position join is no longer safe, this script needs updating")
    df = main.copy()
    df["start_frame"] = frames["start_frame"]
    df["end_frame"] = frames["end_frame"]
    return df


def _put(base: str, timeout: int = 15):
    def call(path, body=None):
        url = path if path.startswith("http") else base + path
        req = urllib.request.Request(url, data=body,
                                     method="PUT" if body is not None else "GET",
                                     headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"Labeler error: HTTP {e.code} — {e.read().decode(errors='replace')}")
        except urllib.error.URLError as e:
            sys.exit(f"Could not reach the labeler at {base} ({e.reason}). Is it running?")
    return call


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX, help="path to asd_session_qc_social.xlsx")
    ap.add_argument("--list", action="store_true", help="print every session row, then exit")
    ap.add_argument("--day", help="e.g. day1")
    ap.add_argument("--cage", help="e.g. cage1RB")
    ap.add_argument("--pid", help="labeler project id to set the window on (omit to just print the row)")
    ap.add_argument("--vid", help="labeler video id within --pid")
    ap.add_argument("--url", default="http://127.0.0.1:8760", help="labeler base URL (default %(default)s)")
    args = ap.parse_args()

    df = load_sessions(args.xlsx)

    if args.list:
        cols = ["day", "cage", "animal_id", "entry_time", "exit_time", "start_frame", "end_frame", "flags"]
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(df[cols].to_string(index=False))
        return

    if not args.day or not args.cage:
        ap.error("give --day and --cage (see --list), or use --list alone")

    rows = df[(df["day"] == args.day) & (df["cage"] == args.cage)]
    if len(rows) == 0:
        sys.exit(f"no row for day={args.day!r} cage={args.cage!r} in {args.xlsx}")
    if len(rows) > 1:
        sys.exit(f"{len(rows)} rows match day={args.day!r} cage={args.cage!r} — expected one")
    row = rows.iloc[0]

    if pd.isna(row["start_frame"]) or pd.isna(row["end_frame"]):
        sys.exit(f"day={args.day} cage={args.cage}: no start_frame/end_frame in the sheet "
                 f"(entry_time={row['entry_time']!r}, exit_time={row['exit_time']!r}) — "
                 f"this looks like a session with no social window, not something to import")

    s, e = int(row["start_frame"]), int(row["end_frame"])
    print(f"day={args.day} cage={args.cage} animal_id={row.get('animal_id')}: "
         f"entry={row['entry_time']:.2f}min exit={row['exit_time']:.2f}min -> frames [{s}, {e})")
    if row.get("flags") and not pd.isna(row.get("flags")):
        print(f"  NOTE flags={row['flags']!r} — check this session before trusting it")

    if not args.pid:
        return
    if not args.vid:
        ap.error("--pid requires --vid (which clip in that project this row applies to)")

    call = _put(args.url.rstrip("/"))
    q_pid, q_vid = urllib.parse.quote(args.pid, safe=""), urllib.parse.quote(args.vid, safe="")
    got = call(f"/api/projects/{q_pid}/videos/{q_vid}/social-window",
              json.dumps({"start_frame": s, "end_frame": e}).encode())
    print(f"set on {args.pid}/{args.vid}: social_start={got['social_start']} social_end={got['social_end']}")


if __name__ == "__main__":
    main()
