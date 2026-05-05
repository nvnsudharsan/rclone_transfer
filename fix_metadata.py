#!/usr/bin/env python3
"""
Fix metadata issues in the UT-GraphCast hindcast NetCDF files.

Four fixes are applied:
  1. Clip negative specific humidity (q) values to 0.
  2. Correct vertical velocity (w) units from "m s-1" / "m/s" to "Pa s-1"
     (and update long_name / standard_name accordingly).
  3. Clip negative 6-hour total precipitation (tp06) values to 0.
  4. Correct geopotential (z) labelling: data is geopotential in m^2 s^-2
     (max ~ 200,000 in upper levels) but was previously labelled as
     geopotential height in m. The fix relabels units / long_name /
     standard_name. THE DATA ARRAY IS NOT TOUCHED.

Design notes:
  * Files are modified in place, but every change appends a CF-style
    timestamped entry to the global ``history`` attribute so the
    edits remain auditable.
  * Large 4-D variables (q especially) are streamed slice-by-slice
    along the leading axis to avoid loading a full ~3 GB array
    into RAM.
  * --dry-run actually inspects every file and reports what *would*
    change, without writing anything.
  * Errors on individual files are logged and counted; processing
    continues so a single bad file does not abort a multi-year run.

Usage examples:
  python fix_metadata.py --dry-run                    # check 1979-2024, no writes
  python fix_metadata.py --years 1979                 # apply fixes to 1979 only
  python fix_metadata.py --years 1979 1985            # 1979-1985 inclusive
  python fix_metadata.py --years 1979 2024 --yes      # full archive, no prompt
  python fix_metadata.py --only q --years 1990 1995   # only fix q for 1990-1995
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from glob import glob
from typing import Optional

import netCDF4 as nc
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_history(ds: nc.Dataset, msg: str) -> None:
    """Append a CF-style timestamped line to the global history attribute."""
    line = f"{_utc_now_iso()}: {msg}"
    existing = getattr(ds, "history", "")
    ds.history = f"{existing}\n{line}" if existing else line


def _stream_negative_count(var: nc.Variable) -> int:
    """Count negative values in a variable by streaming along axis 0.

    This avoids loading the full array into memory. Works for any rank >= 1.
    """
    total = 0
    if var.ndim == 0:
        return 0
    for i in range(var.shape[0]):
        chunk = np.asarray(var[i])
        # NaNs compare False against 0 by default — np.less is safe.
        total += int(np.sum(chunk < 0))
    return total


def _stream_clip_negative(var: nc.Variable) -> int:
    """Clip negatives to 0 in place, slice by slice. Returns count clipped.

    Only writes back slices that actually contain negatives, to minimize I/O.
    """
    total = 0
    for i in range(var.shape[0]):
        chunk = np.asarray(var[i])
        n = int(np.sum(chunk < 0))
        if n > 0:
            total += n
            np.maximum(chunk, 0.0, out=chunk)
            var[i] = chunk
    return total


# ---------------------------------------------------------------------------
# Per-file fix functions
#
# Each returns one of:
#   (True,  n_changed)  -> change applied (or would be applied in dry-run)
#   (False, 0)          -> nothing to do
#   (None,  0)          -> error (already logged)
# ---------------------------------------------------------------------------

def fix_specific_humidity(filepath: str, dry_run: bool = False):
    mode = "r" if dry_run else "r+"
    try:
        with nc.Dataset(filepath, mode) as ds:
            if "q" not in ds.variables:
                return False, 0
            q_var = ds.variables["q"]

            if dry_run:
                neg = _stream_negative_count(q_var)
                return (neg > 0), neg

            neg = _stream_clip_negative(q_var)
            if neg > 0:
                _append_history(ds, f"Clipped {neg} negative q values to 0")
                return True, neg
            return False, 0
    except Exception as e:
        logging.error("q fix failed for %s: %s", os.path.basename(filepath), e)
        return None, 0


def fix_vertical_velocity_units(filepath: str, dry_run: bool = False):
    mode = "r" if dry_run else "r+"
    try:
        with nc.Dataset(filepath, mode) as ds:
            if "w" not in ds.variables:
                return False, 0
            w_var = ds.variables["w"]
            current_units = getattr(w_var, "units", "")

            if current_units not in ("m s-1", "m/s", "m.s-1"):
                return False, 0

            if dry_run:
                return True, 1

            w_var.units = "Pa s-1"
            long_name = getattr(w_var, "long_name", "")
            if "Wind" in long_name or not long_name:
                w_var.long_name = "Vertical Velocity (Omega)"
            w_var.standard_name = "lagrangian_tendency_of_air_pressure"
            _append_history(
                ds,
                f"Corrected w units '{current_units}' -> 'Pa s-1' "
                "and updated long_name/standard_name",
            )
            return True, 1
    except Exception as e:
        logging.error("w fix failed for %s: %s", os.path.basename(filepath), e)
        return None, 0


def fix_precipitation(filepath: str, dry_run: bool = False):
    mode = "r" if dry_run else "r+"
    try:
        with nc.Dataset(filepath, mode) as ds:
            if "tp06" not in ds.variables:
                return False, 0
            tp_var = ds.variables["tp06"]

            if dry_run:
                neg = _stream_negative_count(tp_var)
                return (neg > 0), neg

            neg = _stream_clip_negative(tp_var)
            if neg > 0:
                _append_history(ds, f"Clipped {neg} negative tp06 values to 0")
                return True, neg
            return False, 0
    except Exception as e:
        logging.error("tp06 fix failed for %s: %s", os.path.basename(filepath), e)
        return None, 0


def fix_geopotential_units(filepath: str, dry_run: bool = False):
    """Relabel z from height (m) to geopotential (m**2 s-2). Data unchanged.

    The UT-GraphCast files store geopotential (m^2 s^-2) but were previously
    labelled as geopotential height in m. Values typically reach ~200,000
    in the upper troposphere/stratosphere -- consistent with geopotential,
    not heights. We correct the labels and leave the array alone.
    """
    mode = "r" if dry_run else "r+"
    try:
        with nc.Dataset(filepath, mode) as ds:
            if "z" not in ds.variables:
                return False, 0
            z_var = ds.variables["z"]
            current_units = getattr(z_var, "units", "")

            # Already correct (any geopotential variant) -> nothing to do.
            u_norm = current_units.replace(" ", "").lower()
            if u_norm in ("m2s-2", "m**2s-2", "m^2s-2", "m2/s2"):
                return False, 0
            # Only fix files that are mislabelled as height.
            if current_units != "m":
                return False, 0

            if dry_run:
                return True, 1

            z_var.units = "m**2 s-2"
            z_var.long_name = "Geopotential"
            z_var.standard_name = "geopotential"
            _append_history(
                ds,
                "Relabelled z: units 'm' -> 'm**2 s-2', long_name "
                "'Geopotential Height' -> 'Geopotential', standard_name "
                "'geopotential_height' -> 'geopotential' (values unchanged; "
                "they were always geopotential, not height)",
            )
            return True, 1
    except Exception as e:
        logging.error("z fix failed for %s: %s", os.path.basename(filepath), e)
        return None, 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FIX_REGISTRY = {
    # key: (glob pattern, label, fix function)
    "q":  ("q_pressure_levels_*.nc",  "specific humidity (q)",     fix_specific_humidity),
    "w":  ("w_pressure_levels_*.nc",  "vertical velocity (w)",     fix_vertical_velocity_units),
    "tp": ("surface_variables_*.nc",  "precipitation (tp06)",      fix_precipitation),
    "z":  ("z_pressure_levels_*.nc",  "geopotential (z) units",    fix_geopotential_units),
}


def process_year(year_dir: str, fixes: list[str], dry_run: bool, stats: dict) -> None:
    for key in fixes:
        pattern, label, fn = FIX_REGISTRY[key]
        files = sorted(glob(os.path.join(year_dir, pattern)))
        if not files:
            print(f"  [{label}] no files matched")
            continue
        print(f"  [{label}] {len(files)} files...")
        for i, fp in enumerate(files, 1):
            result, n = fn(fp, dry_run=dry_run)
            if result is True:
                stats[f"{key}_changed"] += 1
                stats[f"{key}_count"]   += n
            elif result is None:
                stats[f"{key}_errors"]  += 1
            if i % 50 == 0 or i == len(files):
                print(f"    progress: {i}/{len(files)}", end="\r")
        print()  # newline after progress


def process_all(base_dir: str, years: range, fixes: list[str], dry_run: bool) -> dict:
    stats = {f"{k}_changed": 0 for k in FIX_REGISTRY}
    stats.update({f"{k}_count":  0 for k in FIX_REGISTRY})
    stats.update({f"{k}_errors": 0 for k in FIX_REGISTRY})

    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.isdir(year_dir):
            print(f"\n[skip] {year_dir} does not exist")
            continue
        print(f"\n{'='*70}\nYear: {year}\n{'='*70}")
        process_year(year_dir, fixes, dry_run, stats)

    return stats


def print_summary(stats: dict, dry_run: bool, duration) -> None:
    verb = "would change" if dry_run else "changed"
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Duration: {duration}")
    # Fixes that report a count of values touched (vs. just relabelling).
    count_keys = {"q", "tp"}
    for key, (_, label, _) in FIX_REGISTRY.items():
        c = stats[f"{key}_changed"]
        n = stats[f"{key}_count"]
        e = stats[f"{key}_errors"]
        print(f"  {label}:")
        print(f"    files {verb}: {c}")
        if key in count_keys:
            print(f"    negative values {'detected' if dry_run else 'clipped'}: {n}")
        print(f"    errors: {e}")
    if dry_run:
        print("\n(DRY RUN - no files were modified)")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fix metadata in UT-GraphCast hindcast NetCDF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-dir",
        default="/scratch/09295/naveens/hindcast",
        help="Root directory containing year subfolders.",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[1979, 2024],
        metavar="YEAR",
        help="One year, or START END (inclusive).",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(FIX_REGISTRY.keys()),
        default=list(FIX_REGISTRY.keys()),
        help="Subset of fixes to apply.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Inspect and report; do not modify files.")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the interactive confirmation prompt.")
    p.add_argument("--log-file", default=None,
                   help="Optional path for an error log.")
    args = p.parse_args(argv)

    if len(args.years) == 1:
        args.start_year = args.end_year = args.years[0]
    elif len(args.years) == 2:
        args.start_year, args.end_year = sorted(args.years)
    else:
        p.error("--years takes 1 or 2 integers")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    log_handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )

    print("=" * 70)
    print("NetCDF METADATA FIX")
    print("=" * 70)
    print(f"Base dir : {args.base_dir}")
    print(f"Years    : {args.start_year}-{args.end_year}")
    print(f"Fixes    : {', '.join(args.only)}")
    print(f"Dry run  : {args.dry_run}")

    if not args.dry_run and not args.yes:
        try:
            ans = input("\nThis will modify files in place. Proceed? (yes/no): ")
        except EOFError:
            print("Non-interactive shell; pass --yes to confirm. Aborted.")
            return 1
        if ans.strip().lower() != "yes":
            print("Aborted.")
            return 1

    start = datetime.now()
    stats = process_all(
        args.base_dir,
        range(args.start_year, args.end_year + 1),
        args.only,
        args.dry_run,
    )
    print_summary(stats, args.dry_run, datetime.now() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
