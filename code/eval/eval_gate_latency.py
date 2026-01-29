#!/usr/bin/env python3
import argparse, sys, glob, os
import pandas as pd

REQ_COLS = {
    "allowed","marker_ok","condition_ok","template_lock_ok",
    "banner_present","latency_ms","case","model","dataset"
}

def load_frames(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        print("No files matched. Check your paths.", file=sys.stderr); sys.exit(1)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}", file=sys.stderr)
    if not dfs:
        print("No readable CSVs.", file=sys.stderr); sys.exit(1)
    return pd.concat(dfs, ignore_index=True)

def gate_metrics(df):
    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    valid   = (df["marker_ok"].astype(bool)
               & df["condition_ok"].astype(bool)
               & df["template_lock_ok"].astype(bool))
    allowed = df["allowed"].astype(bool)
    banner  = df["banner_present"].astype(bool)

    correct_allow     = (allowed == valid)
    false_pos         = (allowed & ~valid)     # allowed though invalid
    false_neg         = (~allowed & valid)     # blocked though valid
    banner_consistent = (banner == allowed)

    out = {
        "n_rows": len(df),
        "gate_accuracy": correct_allow.mean(),
        "gate_fp_rate": false_pos.mean(),
        "gate_fn_rate": false_neg.mean(),
        "banner_consistency_rate": banner_consistent.mean(),
    }
    denom_invalid = (~valid).sum()
    denom_valid   = (valid).sum()
    out["fp_rate_given_invalid"] = false_pos.sum()/denom_invalid if denom_invalid else float("nan")
    out["fn_rate_given_valid"]   = false_neg.sum()/denom_valid   if denom_valid   else float("nan")
    return pd.Series(out)

def latency_metrics(df):
    s = {}
    df = df.copy()
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["latency_ms"])
    if df.empty:
        return pd.Series({
            "lat_mean_all": float("nan"), "lat_p95_all": float("nan"),
            "lat_mean_allowed": float("nan"), "lat_p95_allowed": float("nan"),
            "lat_mean_blocked": float("nan"), "lat_p95_blocked": float("nan"),
            "lat_overhead_mean_abs_ms": float("nan"), "lat_overhead_mean_pct": float("nan"),
            "lat_overhead_p95_abs_ms": float("nan"), "lat_overhead_p95_pct": float("nan"),
        })

    s["lat_mean_all"] = df["latency_ms"].mean()
    s["lat_p95_all"]  = df["latency_ms"].quantile(0.95)

    for key, mask in [("allowed", df["allowed"].astype(bool)),
                      ("blocked", ~df["allowed"].astype(bool))]:
        if mask.any():
            s[f"lat_mean_{key}"] = df.loc[mask, "latency_ms"].mean()
            s[f"lat_p95_{key}"]  = df.loc[mask, "latency_ms"].quantile(0.95)
        else:
            s[f"lat_mean_{key}"] = float("nan")
            s[f"lat_p95_{key}"]  = float("nan")

    base = s["lat_mean_blocked"]
    if base == base and base:  # not NaN and nonzero
        s["lat_overhead_mean_abs_ms"] = s["lat_mean_allowed"] - s["lat_mean_blocked"]
        s["lat_overhead_mean_pct"]    = 100.0 * s["lat_overhead_mean_abs_ms"] / base
    else:
        s["lat_overhead_mean_abs_ms"] = float("nan")
        s["lat_overhead_mean_pct"]    = float("nan")

    base_p95 = s["lat_p95_blocked"]
    if base_p95 == base_p95 and base_p95:
        s["lat_overhead_p95_abs_ms"] = s["lat_p95_allowed"] - s["lat_p95_blocked"]
        s["lat_overhead_p95_pct"]    = 100.0 * s["lat_overhead_p95_abs_ms"] / base_p95
    else:
        s["lat_overhead_p95_abs_ms"] = float("nan")
        s["lat_overhead_p95_pct"]    = float("nan")

    return pd.Series(s)

def summarize(df, group_cols):
    # Keep only required cols if present
    keep = list(REQ_COLS | set(group_cols) | {"latency_ms","__file"})
    df = df[[c for c in df.columns if c in keep]].copy()

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        gm = gate_metrics(g)
        lm = latency_metrics(g)
        row = {}
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys): row[k] = v
        else:
            row[group_cols[0]] = keys
        row.update(gm.to_dict()); row.update(lm.to_dict())
        row["n_rows"] = int(row["n_rows"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Gate correctness and latency overhead evaluator.")
    ap.add_argument("inputs", nargs="+", help="CSV paths or globs (e.g., outputs/**/*.csv)")
    ap.add_argument("--groupby", nargs="*", default=["__file","dataset","model","case"],
                    help="Columns to group by (default: __file,dataset,model,case)")
    ap.add_argument("--out_csv", default="gate_latency_summary.csv", help="Output summary CSV path")
    args = ap.parse_args()

    df = load_frames(args.inputs)
    group_cols = [c for c in args.groupby if c in df.columns or c=="__file"]
    if not group_cols: group_cols = ["__file"]

    summary = summarize(df, group_cols)
    print("\n=== Gate & Latency Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(args.out_csv, index=False)
    print("\nWrote:", args.out_csv)

if __name__ == "__main__":
    main()
