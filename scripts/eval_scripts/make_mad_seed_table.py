#!/usr/bin/env python3
"""
Print the accuracy of each attribute when the caption names it and when it does not, from
the CSVs written by experiments/mad/05_eval_seed_tables.sh (one row per seed).

Usage:
    python scripts/eval_scripts/make_mad_seed_table.py [--rows] <eval.csv> [<eval.csv> ...]

Without --rows, the rows of each file are averaged (mean +- std over the seeds). With --rows,
each row is printed separately.

mgap/utils.py:write_to_csv writes the results after a config prefix of variable length, so
the accuracy block is located by its shape: 18 consecutive floats making 6 groups of
    (present_in_both, wrong_thing_present_in_caption, caption_does_not_contain_it)
where the middle entry is always -1. The digit is always named in the caption.
"""

import csv
import os
import statistics as st
import sys

CATS = ["Thickthinning", "Swelling", "Fracture", "Scaling", "Color", "digit"]
CHANCE = {"Thickthinning": 33.3, "Swelling": 50.0, "Fracture": 50.0,
          "Scaling": 50.0, "Color": 14.3, "digit": 10.0}


def find_block(row):
    """Return the index where the 18-value accuracy block starts, or None."""
    for off in range(len(row) - 17):
        try:
            vals = [float(row[off + k]) for k in range(18)]
        except ValueError:
            continue
        if all(vals[3 * g + 1] == -1.0 for g in range(6)):
            return off
    return None


def seed_label(row):
    """The -seedN token of the mask config name; seed 0 has none."""
    for field in row:
        if field.startswith("seed") and field[4:].isdigit():
            return field
    return "seed0"


def summarise(path, per_row=False):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    if not rows:
        print(f"{path}: empty")
        return

    offs = [find_block(r) for r in rows]
    if any(o is None for o in offs):
        bad = [i for i, o in enumerate(offs) if o is None]
        print(f"{path}: could not locate the accuracy block in row(s) {bad}")
        return

    if per_row:
        for r, o in zip(rows, offs):
            print(f"\n{os.path.basename(path)}  [{seed_label(r)}]")
            print(f"  {'attribute':<14} {'in caption':>10} {'not in caption':>16} {'chance':>8}")
            for i, cat in enumerate(CATS):
                p_v = float(r[o + 3 * i]) * 100
                a_v = float(r[o + 3 * i + 2]) * 100
                a_s = "n/a" if a_v < 0 else f"{a_v:5.1f}"
                print(f"  {cat:<14} {p_v:>10.1f} {a_s:>16} {CHANCE[cat]:>7.1f}")
        return

    print(f"\n{os.path.basename(path)}  ({len(rows)} run(s): {', '.join(seed_label(r) for r in rows)})")
    print(f"  {'attribute':<14} {'in caption':>16} {'not in caption':>18} {'chance':>8}")
    for i, cat in enumerate(CATS):
        pres = [float(r[o + 3 * i]) * 100 for r, o in zip(rows, offs)]
        absent = [float(r[o + 3 * i + 2]) * 100 for r, o in zip(rows, offs)]
        if all(a < 0 for a in absent):
            a_str = "n/a"
        else:
            a_str = f"{st.mean(absent):5.1f}"
            if len(rows) > 1:
                a_str += f" +- {st.stdev(absent):4.1f}"
        p_str = f"{st.mean(pres):5.1f}"
        if len(rows) > 1:
            p_str += f" +- {st.stdev(pres):4.1f}"
        print(f"  {cat:<14} {p_str:>16} {a_str:>18} {CHANCE[cat]:>7.1f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    per_row = "--rows" in argv
    paths = [a for a in argv if a != "--rows"]
    if not paths:
        print(__doc__)
        sys.exit(1)
    for p in paths:
        summarise(p, per_row=per_row)
