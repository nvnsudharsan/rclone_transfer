#!/usr/bin/env python3
"""
Inspect UT-GraphCast hindcast NetCDF files and verify units and value ranges.

Improvements over the original check_metadata.py:
  * Stats (min/max/mean/std) for large 4-D variables are computed by
    streaming along the leading axis, so memory stays bounded (~250 MB
    per slice) even for 13x61x721x1440 pressure-level files.
  * Each variable is checked against an ERA5/CF expected-value table.
    Failures (out-of-range, wrong units, presence of NaNs/negatives where
    none should exist) are flagged with a [FAIL] marker.
  * The geopotential ``z`` ambiguity is resolved automatically:
    if units == "m" the values are validated as geopotential height
    (typical 0-50,000 m); if units include "m2 s-2" / "m**2 s-2" they
    are validated as geopotential proper (typical 0-500,000 m^2 s^-2).
    A unit/value mismatch (e.g. units="m" but max ~ 5e5) is flagged.
  * CLI: pick a single file, the first file per year, or every file in
    a year range. Choose which dataset types to check.

Usage:
  python check_metadata.py                           # default: first file per type, 1979
  python check_metadata.py --years 1979 1985         # first file per type, 1979..1985
  python check_metadata.py --years 1979 --all        # every file in 1979
  python check_metadata.py --file path/to/file.nc    # one specific file
  python check_metadata.py --only q t z              # only those datasets
  python check_metadata.py --quiet                   # suppress per-variable detail
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from typing import Optional

import netCDF4 as nc
import numpy as np


# ---------------------------------------------------------------------------
# Expected unit/value rules
# Each rule is (acceptable_units_set_or_None, value_validator_or_None,
#               allow_negative, description).
# A value_validator is a callable (vmin, vmax) -> list[str] of failure messages,
# or None to skip range checking.
# ---------------------------------------------------------------------------

def _range_check(lo: float, hi: float, name: str):
    def check(vmin, vmax):
        msgs = []
        if vmin < lo:
            msgs.append(f"{name} min {vmin:.4g} < {lo}")
        if vmax > hi:
            msgs.append(f"{name} max {vmax:.4g} > {hi}")
        return msgs
    return check


def _z_check(units: str):
    """Geopotential vs geopotential-height auto-detector."""
    u = units.replace(" ", "").lower()
    if u == "m":
        return _range_check(-500, 50_000, "geopotential height (m)")
    if u in ("m2s-2", "m**2s-2", "m2/s2", "m^2s-2"):
        return _range_check(-5_000, 500_000, "geopotential (m^2 s^-2)")
    return None  # unknown units -> caller flags separately

VAR_RULES = {
    # surface
    "u10m": (["m s-1", "m/s"], _range_check(-150, 150, "u10m"), True),
    "v10m": (["m s-1", "m/s"], _range_check(-150, 150, "v10m"), True),
    "t2m":  (["K"],             _range_check(180, 340, "t2m"),  False),
    "msl":  (["Pa"],             _range_check(80_000, 110_000, "msl"), False),
    "tp06": (["m"],              _range_check(0, 1.0, "tp06"),  False),
    # pressure-level
    "t":    (["K"],             _range_check(150, 340, "t"),    False),
    "q":    (["kg kg-1", "1"],  _range_check(0, 0.05, "q"),     False),
    "u":    (["m s-1", "m/s"], _range_check(-200, 200, "u"),    True),
    "v":    (["m s-1", "m/s"], _range_check(-200, 200, "v"),    True),
    "w":    (["Pa s-1", "Pa/s"], _range_check(-50, 50, "w"),    True),
    "z":    (None, None, True),  # handled specially below
}


# ---------------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------------

def streaming_stats(var: nc.Variable):
    """Return (vmin, vmax, mean, std, n_valid, n_nan, n_negative).

    Streams along axis 0; accumulates sums in float64.
    Works for variables of rank >= 1.
    """
    vmin = np.inf
    vmax = -np.inf
    s = 0.0
    s2 = 0.0
    n = 0
    n_nan = 0
    n_neg = 0

    if var.ndim == 0:
        val = float(var[...])
        return val, val, val, 0.0, 1, 0, int(val < 0)

    for i in range(var.shape[0]):
        chunk = np.asarray(var[i])
        # Treat masked / NaN consistently
        if np.ma.isMaskedArray(chunk):
            chunk = chunk.filled(np.nan)
        finite = np.isfinite(chunk)
        n_nan += int(chunk.size - finite.sum())
        if not finite.any():
            continue
        vals = chunk[finite].astype(np.float64, copy=False)
        vmin = min(vmin, float(vals.min()))
        vmax = max(vmax, float(vals.max()))
        s   += float(vals.sum())
        s2  += float(np.dot(vals, vals))
        n   += vals.size
        n_neg += int((vals < 0).sum())

    if n == 0:
        return None, None, None, None, 0, n_nan, 0
    mean = s / n
    var_ = max(s2 / n - mean * mean, 0.0)  # guard against -1e-12 noise
    std = float(np.sqrt(var_))
    return vmin, vmax, mean, std, n, n_nan, n_neg


# ---------------------------------------------------------------------------
# Per-variable inspection
# ---------------------------------------------------------------------------

def inspect_variable(var: nc.Variable, name: str, quiet: bool):
    """Print info about a single variable. Return list of failure messages."""
    failures: list[str] = []
    units = getattr(var, "units", "")
    long_name = getattr(var, "long_name", "")
    std_name = getattr(var, "standard_name", "")

    if not quiet:
        print(f"\n  {name}:")
        print(f"    Shape: {var.shape}    Type: {var.dtype}")
        if units:     print(f"    Units: {units}")
        if long_name: print(f"    Long name: {long_name}")
        if std_name:  print(f"    Standard name: {std_name}")

    # 1-D coordinate variables: print values/range and stop here.
    if var.ndim <= 1:
        if not quiet:
            data = np.asarray(var[:])
            if data.size <= 20:
                print(f"    Values: {data}")
            else:
                print(f"    Range: [{data.min()}, {data.max()}]")
        return failures

    # Multi-D data variable: streaming stats.
    vmin, vmax, mean, std, n, n_nan, n_neg = streaming_stats(var)
    if n == 0:
        failures.append(f"{name}: all values are NaN/missing")
        if not quiet:
            print("    [FAIL] all values NaN/missing")
        return failures

    if not quiet:
        print(f"    Value range: [{vmin:.6e}, {vmax:.6e}]")
        print(f"    Mean: {mean:.6e}    Std: {std:.6e}")
        print(f"    Valid: {n}    NaN: {n_nan}    Negatives: {n_neg}")

    # Apply rules.
    if name in VAR_RULES:
        allowed_units, range_fn, allow_neg = VAR_RULES[name]

        # Special-case z: pick range_fn based on declared units, and prefer
        # the "labelled wrong" diagnosis over the generic out-of-range message.
        if name == "z":
            u_norm = units.replace(" ", "").lower()
            mislabel = None
            if units == "m" and vmax > 50_000:
                mislabel = (
                    f"z labelled units='m' but max={vmax:.4g} looks like "
                    "geopotential (m^2 s^-2); divide by 9.80665 or fix units"
                )
            elif u_norm in ("m2s-2", "m**2s-2", "m^2s-2", "m2/s2") and vmax < 50_000:
                mislabel = (
                    f"z labelled m^2 s^-2 but max={vmax:.4g} looks like "
                    "height (m); multiply by 9.80665 or fix units"
                )

            if mislabel is not None:
                failures.append(mislabel)
                range_fn = None  # don't double-report
            else:
                range_fn = _z_check(units)
                if range_fn is None:
                    failures.append(
                        f"z has unrecognized units '{units}' -- expected "
                        "'m' (geopotential height) or 'm2 s-2' (geopotential)"
                    )
            allowed_units = ["m", "m2 s-2", "m**2 s-2", "m^2 s-2"]

        if allowed_units is not None and units not in allowed_units:
            failures.append(
                f"{name} units '{units}' not in expected {allowed_units}"
            )
        if range_fn is not None:
            failures.extend(range_fn(vmin, vmax))
        if not allow_neg and n_neg > 0:
            failures.append(f"{name} has {n_neg} negative values (none allowed)")
    else:
        # Unknown variable: just note it, don't fail.
        pass

    if failures:
        # Always surface failures, even in quiet mode.
        if quiet:
            print(f"  {name}: [FAIL] " + "; ".join(failures))
        else:
            for msg in failures:
                print(f"    [FAIL] {msg}")
    return failures


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def inspect_file(filepath: str, label: str, quiet: bool) -> tuple[int, int]:
    """Return (n_failures, n_variables_checked)."""
    print(f"\n{'='*78}")
    print(f"{label}")
    print(f"  File: {filepath}")
    print(f"{'='*78}")

    n_fail = 0
    n_var = 0
    try:
        with nc.Dataset(filepath, "r") as ds:
            if not quiet:
                print("\nDIMENSIONS:")
                for d, dim in ds.dimensions.items():
                    print(f"  {d}: {len(dim)}")
                print("\nVARIABLES:")
            for name, var in ds.variables.items():
                fails = inspect_variable(var, name, quiet)
                n_fail += len(fails)
                n_var += 1
            if not quiet:
                print("\nGLOBAL ATTRIBUTES:")
                for a in ds.ncattrs():
                    val = getattr(ds, a)
                    # Truncate verbose 'history' attr
                    if a == "history" and isinstance(val, str) and len(val) > 400:
                        val = val[:200] + " ... " + val[-200:]
                    print(f"  {a}: {val}")
    except Exception as e:
        print(f"\n[FAIL] could not open: {e}")
        return 1, 0

    if quiet and n_fail == 0:
        print("  OK")
    elif n_fail > 0:
        print(f"\n  >>> {n_fail} failure(s) in this file <<<")
    return n_fail, n_var


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

DATASET_PATTERNS = {
    "surface": ("surface_variables_*.nc", "surface variables"),
    "q":       ("q_pressure_levels_*.nc", "specific humidity (q)"),
    "t":       ("t_pressure_levels_*.nc", "temperature (t)"),
    "u":       ("u_pressure_levels_*.nc", "u wind (u)"),
    "v":       ("v_pressure_levels_*.nc", "v wind (v)"),
    "w":       ("w_pressure_levels_*.nc", "vertical velocity (w)"),
    "z":       ("z_pressure_levels_*.nc", "geopotential (z)"),
}


def gather_files(base_dir: str, years: range, only: list[str], all_files: bool):
    """Yield (label, filepath) tuples to inspect."""
    for year in years:
        ydir = os.path.join(base_dir, str(year))
        if not os.path.isdir(ydir):
            print(f"[skip] {ydir} not found")
            continue
        for key in only:
            pattern, label = DATASET_PATTERNS[key]
            matches = sorted(glob(os.path.join(ydir, pattern)))
            if not matches:
                print(f"[skip] no {pattern} in {ydir}")
                continue
            files = matches if all_files else matches[:1]
            for fp in files:
                yield f"{year} / {label}", fp


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check UT-GraphCast hindcast NetCDF metadata and value ranges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir", default="/scratch/09295/naveens/hindcast")
    p.add_argument("--years", nargs="+", type=int, default=[1979],
                   metavar="YEAR", help="One year, or START END (inclusive).")
    p.add_argument("--only", nargs="+", choices=list(DATASET_PATTERNS.keys()),
                   default=list(DATASET_PATTERNS.keys()),
                   help="Subset of dataset types to inspect.")
    p.add_argument("--file", default=None,
                   help="Inspect a single file (bypasses --base-dir/--years).")
    p.add_argument("--all", action="store_true",
                   help="Inspect every matching file (default: just the first per type per year).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-variable detail; only print failures and summary.")
    args = p.parse_args(argv)
    if len(args.years) == 1:
        args.start_year = args.end_year = args.years[0]
    elif len(args.years) == 2:
        args.start_year, args.end_year = sorted(args.years)
    else:
        p.error("--years takes 1 or 2 integers")
    return args


REFERENCE_TEXT = """
EXPECTED UNITS / RANGES (ERA5 standard)

Surface:
  u10m, v10m  : m s-1     -150..150
  t2m         : K          180..340
  msl         : Pa         80000..110000
  tp06        : m          0..1

Pressure-level:
  t           : K          150..340
  q           : kg kg-1    0..0.05      (no negatives)
  u, v        : m s-1     -200..200
  w           : Pa s-1    -50..50
  z           : m          0..50000     (geopotential height)
                or m2 s-2  0..500000    (geopotential)
  plevel      : hPa        50..1000
"""


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    print("="*78)
    print("UT-GRAPHCAST HINDCAST METADATA CHECK")
    print("="*78)

    total_fail = 0
    total_files = 0
    bad_files: list[tuple[str, int]] = []

    if args.file:
        n_fail, _ = inspect_file(args.file, "single file", args.quiet)
        total_fail += n_fail
        total_files += 1
        if n_fail:
            bad_files.append((args.file, n_fail))
    else:
        years = range(args.start_year, args.end_year + 1)
        for label, fp in gather_files(args.base_dir, years, args.only, args.all):
            n_fail, _ = inspect_file(fp, label, args.quiet)
            total_fail += n_fail
            total_files += 1
            if n_fail:
                bad_files.append((fp, n_fail))

    print("\n" + "="*78)
    print("CHECK COMPLETE")
    print("="*78)
    print(f"Files inspected: {total_files}")
    print(f"Total failures : {total_fail}")
    if bad_files:
        print("\nFiles with failures:")
        for fp, n in bad_files:
            print(f"  [{n:3d}]  {fp}")
    print(REFERENCE_TEXT)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())

