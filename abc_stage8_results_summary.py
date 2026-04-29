"""
UrbanAlign 2.0 — Stage 8: Results Summary & Export
===================================================
纯读取 + 统计汇总。从 Stage 1/3/4/5/6/7 的输出文件中提取所有
论文正文和附录需要的表格数据，一键输出 CSV + 控制台格式化表格。

Usage:
    python abc_stage8_results_summary.py
"""

import os
import ast
import json
import warnings

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, CATEGORIES,
    get_stage1_dimensions, get_stage3_output, get_stage4_output,
    get_stage5_output, get_stage6_summary, get_stage7_output,
    get_stage8_output,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _safe_read(path, label=""):
    """Read a CSV with warning on missing file. Returns None if absent."""
    if not os.path.exists(path):
        print(f"  [WARN] 缺少文件: {os.path.basename(path)}  ({label})")
        return None
    return pd.read_csv(path)


def _safe_json(path, label=""):
    """Read a JSON file. Returns None if absent."""
    if not os.path.exists(path):
        print(f"  [WARN] 缺少文件: {os.path.basename(path)}  ({label})")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(df, name):
    """Save DataFrame and report path."""
    out = get_stage8_output(name)
    df.to_csv(out, index=False)
    print(f"  -> 已保存: {os.path.basename(out)}  ({len(df)} rows)")
    return out


def _fmt_pct(v):
    """Format a proportion as percentage string."""
    if pd.isna(v):
        return "  —  "
    return f"{v * 100:5.1f}%"


def _fmt_kappa(v):
    if pd.isna(v):
        return "  —  "
    return f"{v:6.3f}"


def print_formatted_table(title, df, key_cols=None, max_rows=60):
    """Pretty-print a DataFrame section to console."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")
    if df is None or df.empty:
        print("  (无数据)")
        return
    display = df.head(max_rows) if len(df) > max_rows else df
    col_subset = key_cols if key_cols else list(df.columns)
    col_subset = [c for c in col_subset if c in df.columns]
    print(display[col_subset].to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... ({len(df) - max_rows} more rows)")


# ══════════════════════════════════════════════════════════
# Collectors — each returns a DataFrame (or None)
# ══════════════════════════════════════════════════════════

def collect_table1_main_results():
    """
    Table 1: Baselines (C0-C3) + UrbanAlign Mode4+LWRR
    每类 × (5 methods) × (excl/incl) → acc, kappa
    Source: stage4_all_modes_comparison_{cat}.csv
    """
    rows = []
    for cat in CATEGORIES:
        df = _safe_read(get_stage4_output("all_modes_comparison", cat), f"Table1 {cat}")
        if df is None:
            continue

        # Baselines (stage == 'Baseline')
        baselines = df[df["stage"] == "Baseline"]
        for _, r in baselines.iterrows():
            rows.append({
                "category": cat,
                "method": r["experiment"],
                "exclude_equal": r["exclude_equal"],
                "n_samples": int(r["n_samples"]),
                "accuracy": r["accuracy"],
                "kappa": r["kappa"],
                "f1_macro": r.get("f1_macro", np.nan),
            })

        # UrbanAlign Mode 4 + LWRR (Stage3)
        ua = df[(df["stage"] == "Stage3") & (df["mode"] == 4.0)]
        for _, r in ua.iterrows():
            rows.append({
                "category": cat,
                "method": "UrbanAlign-Mode4-LWRR",
                "exclude_equal": r["exclude_equal"],
                "n_samples": int(r["n_samples"]),
                "accuracy": r["accuracy"],
                "kappa": r["kappa"],
                "f1_macro": r.get("f1_macro", np.nan),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_table2a_factorial():
    """
    Table 2a: 4 modes × Stage2/Stage3 factorial ablation (6类)
    Source: stage4_all_modes_comparison_{cat}.csv
    """
    rows = []
    for cat in CATEGORIES:
        df = _safe_read(get_stage4_output("all_modes_comparison", cat), f"Table2a {cat}")
        if df is None:
            continue
        ua = df[df["stage"].isin(["Stage2", "Stage3"])]
        for _, r in ua.iterrows():
            rows.append({
                "category": cat,
                "mode": int(r["mode"]) if pd.notna(r["mode"]) else None,
                "stage": r["stage"],
                "exclude_equal": r["exclude_equal"],
                "n_samples": int(r["n_samples"]),
                "accuracy": r["accuracy"],
                "kappa": r["kappa"],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_table2b_vrm_gain():
    """
    Table 2b: LWRR calibration gain — Mode4 Raw(Stage2) vs Aligned(Stage3)
    Source: stage4_all_modes_comparison_{cat}.csv
    """
    rows = []
    for cat in CATEGORIES:
        df = _safe_read(get_stage4_output("all_modes_comparison", cat), f"Table2b {cat}")
        if df is None:
            continue
        for excl in [True, False]:
            raw = df[(df["stage"] == "Stage2") & (df["mode"] == 4.0) & (df["exclude_equal"] == excl)]
            aln = df[(df["stage"] == "Stage3") & (df["mode"] == 4.0) & (df["exclude_equal"] == excl)]
            if raw.empty or aln.empty:
                continue
            r, a = raw.iloc[0], aln.iloc[0]
            rows.append({
                "category": cat,
                "exclude_equal": excl,
                "raw_accuracy": r["accuracy"],
                "aligned_accuracy": a["accuracy"],
                "delta_pp": (a["accuracy"] - r["accuracy"]) * 100,
                "raw_kappa": r["kappa"],
                "aligned_kappa": a["kappa"],
                "n_samples": int(a["n_samples"]),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_table3_dimension_disc():
    """
    Table 3: Dimension discriminability top-3 per category (Mode 4)
    Source: stage4_dimension_discriminability_{cat}.csv
    """
    rows = []
    for cat in CATEGORIES:
        df = _safe_read(get_stage4_output("dimension_discriminability", cat), f"Table3 {cat}")
        if df is None:
            continue
        # Filter Mode 4 only
        df4 = df[df["mode"] == 4].copy()
        if df4.empty:
            continue
        df4 = df4.sort_values("discriminative_power", ascending=False)
        avg_power = df4["discriminative_power"].mean()
        n_total = int(df4.iloc[0].get("n_samples", 0))
        for rank, (_, r) in enumerate(df4.head(3).iterrows(), 1):
            rows.append({
                "category": cat,
                "rank": rank,
                "dimension": r["dimension"],
                "discriminative_power": r["discriminative_power"],
                "mean_delta": r["mean_delta"],
                "std_delta": r["std_delta"],
                "n_samples": n_total,
                "avg_power": avg_power,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_table4_sensitivity():
    """
    Table 4: Main sensitivity params (K_MAX, ALPHA_HYBRID, SELECTION_RATIO)
    Source: stage5_sensitivity_mode4_{cat}.csv
    """
    target_params = ["K_MAX_ST3", "ALPHA_HYBRID", "SELECTION_RATIO"]
    return _collect_sensitivity_by_params(target_params)


def _collect_sensitivity_by_params(param_list):
    """Generic: collect stage5 rows for specified param names across all cats."""
    rows = []
    for cat in CATEGORIES:
        df = _safe_read(get_stage5_output(4, cat), f"sensitivity {cat}")
        if df is None:
            continue
        sub = df[df["param"].isin(param_list)].copy()
        sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.sort_values(["param", "value_num"])
        for _, r in sub.iterrows():
            rows.append({
                "category": cat,
                "param": r["param"],
                "value": r["value"],
                "accuracy_incl": r["accuracy"],
                "kappa_incl": r["kappa"],
                "accuracy_excl": r.get("accuracy_no_equal", np.nan),
                "kappa_excl": r.get("kappa_no_equal", np.nan),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_supp_param_sweep(param_name):
    """Supplementary: single-param sweep across all categories."""
    return _collect_sensitivity_by_params([param_name])


def collect_supp_combined_best():
    """
    Supplementary: Combined search best configs + CI
    Source: stage5_sensitivity_mode4_{cat}.csv, param='COMBINED'
    """
    rows = []
    for cat in CATEGORIES:
        df = _safe_read(get_stage5_output(4, cat), f"combined {cat}")
        if df is None:
            continue
        comb = df[df["param"] == "COMBINED"].copy()
        if comb.empty:
            continue
        # Sort by excl accuracy descending
        comb["acc_excl"] = comb["accuracy_no_equal"]
        comb = comb.sort_values("acc_excl", ascending=False)
        best = comb.iloc[0]
        rows.append({
            "category": cat,
            "config": best["value"],
            "accuracy_excl": best.get("accuracy_no_equal", np.nan),
            "kappa_excl": best.get("kappa_no_equal", np.nan),
            "accuracy_incl": best["accuracy"],
            "kappa_incl": best["kappa"],
            "acc_ci_low": best.get("acc_ci_low", np.nan),
            "acc_ci_high": best.get("acc_ci_high", np.nan),
            "kappa_ci_low": best.get("kappa_ci_low", np.nan),
            "kappa_ci_high": best.get("kappa_ci_high", np.nan),
            "n_samples": best.get("n_samples", np.nan),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_supp_lwrr_weights():
    """
    Supplementary: LWRR weight statistics per dimension per category
    Source: stage3_mode4_aligned_{cat}.csv (local_weights)
          + stage1_semantic_dimensions_{cat}.json (dimension names)
    """
    rows = []
    for cat in CATEGORIES:
        # Dimension names
        dims_data = _safe_json(get_stage1_dimensions(cat), f"LWRR dims {cat}")
        if dims_data is None:
            continue
        dim_names = [d["name"] for d in dims_data[cat]["dimensions"]]

        # Weights from stage3
        df = _safe_read(get_stage3_output(4, cat), f"LWRR weights {cat}")
        if df is None or "local_weights" not in df.columns:
            continue

        weights_list = []
        for w_str in df["local_weights"].dropna():
            try:
                w = ast.literal_eval(str(w_str))
                if isinstance(w, (list, tuple)) and len(w) == len(dim_names):
                    weights_list.append(w)
            except Exception:
                pass

        if not weights_list:
            print(f"  [WARN] {cat}: 无有效 local_weights")
            continue

        W = np.array(weights_list)
        for i, name in enumerate(dim_names):
            mean_w = W[:, i].mean()
            std_w = W[:, i].std()
            cv = std_w / abs(mean_w) if abs(mean_w) > 0.001 else np.nan
            rows.append({
                "category": cat,
                "dimension": name,
                "mean_weight": mean_w,
                "std_weight": std_w,
                "cv": cv,
                "n_samples": len(W),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_supp_e2e():
    """
    Supplementary: E2E dimension optimization trials
    Source: stage6_e2e_summary_{cat}.csv (all identical across cats)
    """
    # All per-category files are identical; just read one that exists
    for cat in CATEGORIES:
        df = _safe_read(get_stage6_summary(cat), f"E2E {cat}")
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()


def collect_supp_e2e_best_per_cat():
    """
    Supplementary: E2E best trial per category
    Derived from per-category accuracy columns in stage6_e2e_summary.
    """
    df_e2e = collect_supp_e2e()
    if df_e2e.empty:
        return pd.DataFrame()

    rows = []
    for cat in CATEGORIES:
        acc_col = f"acc_{cat}"
        kappa_col = f"kappa_{cat}"
        if acc_col not in df_e2e.columns:
            continue
        valid = df_e2e.dropna(subset=[acc_col])
        if valid.empty:
            continue
        best_idx = valid[acc_col].idxmax()
        best = valid.loc[best_idx]
        rows.append({
            "category": cat,
            "best_trial": int(best["trial"]),
            "phase": best.get("phase", ""),
            "temperature": best.get("temperature", np.nan),
            "accuracy": best[acc_col],
            "kappa": best.get(kappa_col, np.nan),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_best_params():
    """
    Summary: Stage 5 best params per category
    Source: stage5_best_params_{cat}.json
    """
    rows = []
    for cat in CATEGORIES:
        path = os.path.join(OUTPUT_DIR, f"stage5_best_params_{cat}.json")
        data = _safe_json(path, f"best params {cat}")
        if data is None:
            continue
        params = data.get("params", {})
        row = {"category": cat, "source": data.get("source", "")}
        row.update(params)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════
# Key summary numbers (Abstract / Introduction / Conclusion)
# ══════════════════════════════════════════════════════════

def print_key_numbers(t1, t2b):
    """Print headline numbers for easy paper reference."""
    print(f"\n{'=' * 80}")
    print("  关键数值汇总 (Abstract / Introduction / Conclusion)")
    print(f"{'=' * 80}")

    if t1 is None or t1.empty:
        print("  (Table 1 无数据)")
        return

    # UrbanAlign Mode4 LWRR, excl equal
    ua_excl = t1[(t1["method"] == "UrbanAlign-Mode4-LWRR") & (t1["exclude_equal"] == True)]
    ua_incl = t1[(t1["method"] == "UrbanAlign-Mode4-LWRR") & (t1["exclude_equal"] == False)]

    if not ua_excl.empty:
        avg_acc = ua_excl["accuracy"].mean()
        avg_kap = ua_excl["kappa"].mean()
        best_cat = ua_excl.loc[ua_excl["accuracy"].idxmax()]
        worst_cat = ua_excl.loc[ua_excl["accuracy"].idxmin()]
        print(f"\n  UrbanAlign Mode4+LWRR (excl equal):")
        print(f"    平均 Accuracy = {avg_acc:.1%}    平均 Kappa = {avg_kap:.3f}")
        print(f"    最优类别: {best_cat['category']} = {best_cat['accuracy']:.1%}")
        print(f"    最差类别: {worst_cat['category']} = {worst_cat['accuracy']:.1%}")
        for _, r in ua_excl.iterrows():
            print(f"      {r['category']:12s}: Acc={r['accuracy']:.1%}  κ={r['kappa']:.3f}  n={int(r['n_samples'])}")

    if not ua_incl.empty:
        avg_acc_incl = ua_incl["accuracy"].mean()
        avg_kap_incl = ua_incl["kappa"].mean()
        print(f"\n  UrbanAlign Mode4+LWRR (incl equal):")
        print(f"    平均 Accuracy = {avg_acc_incl:.1%}    平均 Kappa = {avg_kap_incl:.3f}")

    # vs Baselines (excl equal)
    baselines = ["Baseline-C0-ResNet", "Baseline-C1-Siamese", "Baseline-C2-SegReg", "Baseline-C3-ZeroShot"]
    for bname in baselines:
        b_excl = t1[(t1["method"] == bname) & (t1["exclude_equal"] == True)]
        if b_excl.empty:
            continue
        b_avg = b_excl["accuracy"].mean()
        delta = (ua_excl["accuracy"].mean() - b_avg) * 100 if not ua_excl.empty else 0
        print(f"    vs {bname}: avg={b_avg:.1%}  (UrbanAlign +{delta:+.1f}pp)")

    # VRM gain
    if t2b is not None and not t2b.empty:
        vrm_excl = t2b[t2b["exclude_equal"] == True]
        if not vrm_excl.empty:
            avg_delta = vrm_excl["delta_pp"].mean()
            print(f"\n  LWRR校准增益 (excl equal):")
            print(f"    平均 Δ = {avg_delta:+.1f}pp")
            for _, r in vrm_excl.iterrows():
                print(f"      {r['category']:12s}: {r['raw_accuracy']:.1%} → {r['aligned_accuracy']:.1%}  ({r['delta_pp']:+.1f}pp)")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  UrbanAlign 2.0 — Stage 8: Results Summary & Export")
    print("=" * 80)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  类别: {', '.join(CATEGORIES)}")

    # ── 1. Table 1: Main Results ──
    print("\n[1/16] Table 1: Main Results (Baselines + UrbanAlign)")
    t1 = collect_table1_main_results()
    if not t1.empty:
        _save(t1, "table1_main_results")
    print_formatted_table("Table 1: Main Results",
                          t1, ["category", "method", "exclude_equal", "accuracy", "kappa", "n_samples"])

    # ── 2. Table 2a: Factorial Ablation ──
    print("\n[2/16] Table 2a: Factorial Ablation (4 modes × 2 stages)")
    t2a = collect_table2a_factorial()
    if not t2a.empty:
        _save(t2a, "table2a_factorial_ablation")
    print_formatted_table("Table 2a: Factorial Ablation",
                          t2a, ["category", "mode", "stage", "exclude_equal", "accuracy", "kappa"])

    # ── 3. Table 2b: VRM Gain ──
    print("\n[3/16] Table 2b: LWRR Calibration Gain")
    t2b = collect_table2b_vrm_gain()
    if not t2b.empty:
        _save(t2b, "table2b_vrm_gain")
    print_formatted_table("Table 2b: LWRR Calibration Gain",
                          t2b, ["category", "exclude_equal", "raw_accuracy", "aligned_accuracy",
                                "delta_pp", "raw_kappa", "aligned_kappa"])

    # ── 4. Table 3: Dimension Discriminability ──
    print("\n[4/16] Table 3: Dimension Discriminability (top-3, Mode 4)")
    t3 = collect_table3_dimension_disc()
    if not t3.empty:
        _save(t3, "table3_dimension_disc")
    print_formatted_table("Table 3: Dimension Discriminability",
                          t3, ["category", "rank", "dimension", "discriminative_power", "avg_power"])

    # ── 5. Table 4: Sensitivity (main 3 params) ──
    print("\n[5/16] Table 4: Sensitivity (K_MAX, ALPHA, SELECTION_RATIO)")
    t4 = collect_table4_sensitivity()
    if not t4.empty:
        _save(t4, "table4_sensitivity")
    print_formatted_table("Table 4: Sensitivity",
                          t4, ["category", "param", "value", "accuracy_excl", "kappa_excl"])

    # ── 6–11. Supplementary param sweeps ──
    supp_params = [
        (6,  "TAU_KERNEL_ST3",        "supp_tau_sweep",       "Tab C3: TAU sweep"),
        (7,  "RIDGE_ALPHA_ST3",       "supp_ridge_sweep",     "Tab C4: RIDGE sweep"),
        (8,  "EQUAL_EPS_ST3",         "supp_eps_sweep",       "Tab C5: EQUAL_EPS sweep"),
        (9,  "EQUAL_CONSENSUS_MIN",   "supp_consensus_sweep", "Tab C5b: CONSENSUS sweep"),
        (10, "ST2_INTENSITY_SIG_THRESH", "supp_threshold_sweep", "Tab A1: Threshold sweep"),
        (11, "LABELED_SET_RATIO",     "supp_ratio_sweep",     "LABELED_SET_RATIO sweep"),
    ]
    for idx, param, fname, title in supp_params:
        print(f"\n[{idx}/16] {title}")
        df = collect_supp_param_sweep(param)
        if not df.empty:
            _save(df, fname)
        print_formatted_table(title, df, ["category", "param", "value", "accuracy_excl", "kappa_excl"])

    # ── 12. Combined Best ──
    print("\n[12/16] Combined Search Best")
    t_comb = collect_supp_combined_best()
    if not t_comb.empty:
        _save(t_comb, "supp_combined_best")
    print_formatted_table("Combined Search Best",
                          t_comb, ["category", "accuracy_excl", "kappa_excl",
                                   "acc_ci_low", "acc_ci_high", "config"])

    # ── 13. LWRR Weights ──
    print("\n[13/16] LWRR Weight Statistics")
    t_wt = collect_supp_lwrr_weights()
    if not t_wt.empty:
        _save(t_wt, "supp_lwrr_weights")
    print_formatted_table("LWRR Weight Statistics",
                          t_wt, ["category", "dimension", "mean_weight", "std_weight", "cv"])

    # ── 14. E2E Trials ──
    print("\n[14/16] E2E Dimension Optimization Trials")
    t_e2e = collect_supp_e2e()
    if not t_e2e.empty:
        _save(t_e2e, "supp_e2e_trials")
    print_formatted_table("E2E Trials", t_e2e,
                          ["trial", "phase", "temperature", "accuracy", "kappa"])

    # ── 15. E2E Best Per Category ──
    print("\n[15/16] E2E Best Trial Per Category")
    t_e2e_best = collect_supp_e2e_best_per_cat()
    if not t_e2e_best.empty:
        _save(t_e2e_best, "supp_e2e_best_per_cat")
    print_formatted_table("E2E Best Per Category", t_e2e_best,
                          ["category", "best_trial", "phase", "temperature", "accuracy", "kappa"])

    # ── 16. Best Params Summary ──
    print("\n[16/16] Stage 5 Best Params Summary")
    t_bp = collect_best_params()
    if not t_bp.empty:
        _save(t_bp, "best_params_summary")
    print_formatted_table("Stage 5 Best Params Summary", t_bp)

    # ── Key numbers ──
    print_key_numbers(t1, t2b)

    print(f"\n{'=' * 80}")
    print("  Stage 8 完成！所有汇总 CSV 已保存到 urbanalign_outputs/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
