#!/usr/bin/env python3
"""
Phase 6 Reproducible Benchmark Plotting Script
Reads results/benchmark_summary.csv and generates performance and load-balance plots:
  1. results/runtime_vs_threads.png
  2. results/speedup_vs_threads.png
  3. results/efficiency_vs_threads.png
  4. results/work_imbalance_and_stages.png
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_STYLES = {
    "Sequential": {"color": "#222222", "marker": "s", "linestyle": "--", "label": "Sequential (1T Ref)"},
    "Static":     {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "OpenMP Static"},
    "Dynamic":    {"color": "#ff7f0e", "marker": "^", "linestyle": "-",  "label": "OpenMP Dynamic"},
    "Guided":     {"color": "#2ca02c", "marker": "D", "linestyle": "-",  "label": "OpenMP Guided"},
    "Adaptive":   {"color": "#d62728", "marker": "P", "linestyle": "-",  "label": "Adaptive (Heavy-Bucket Split)"},
}


def load_summary_csv(csv_path):
    rows = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["dataset_size"] = int(r["dataset_size"])
            r["configured_threads"] = int(r["configured_threads"])
            r["actual_threads"] = int(r["actual_threads"])
            r["participating_threads"] = int(r["participating_threads"])
            r["oversized_buckets"] = int(r["oversized_buckets"])
            r["heavy_bucket_chunks"] = int(r["heavy_bucket_chunks"])
            for float_col in [
                "median_total_ms", "iqr_total_ms", "sd_total_ms",
                "median_fingerprinting_ms", "median_bucket_construction_ms",
                "median_candidate_gen_total_ms", "median_adaptive_decision_ms",
                "median_ordinary_candidate_gen_ms", "median_heavy_chunk_gen_ms",
                "median_heavy_pair_proc_ms", "median_merge_sort_dedup_ms",
                "median_verification_ms", "speedup_total", "efficiency_total",
                "speedup_candidate_gen", "efficiency_candidate_gen",
                "work_imbalance_ratio", "bucket_work_cv"
            ]:
                r[float_col] = float(r[float_col])
            rows.append(r)
    return rows


def filter_rows(rows, dataset_size, profile, policy=None):
    out = [
        r for r in rows
        if r["dataset_size"] == dataset_size and r["profile"] == profile
        and (policy is None or r["policy"] == policy)
    ]
    out.sort(key=lambda x: x["configured_threads"])
    return out


def plot_runtime_vs_threads(rows, out_dir):
    sizes = sorted(list({r["dataset_size"] for r in rows}))
    target_sizes = [s for s in [1000, 2000, 4000, 8000] if s in sizes]
    if not target_sizes:
        target_sizes = sizes

    fig, axes = plt.subplots(len(target_sizes), 2, figsize=(13, 3.4 * len(target_sizes)), sharex=True)
    if len(target_sizes) == 1:
        axes = np.array([axes])

    for row_idx, dsize in enumerate(target_sizes):
        for col_idx, profile in enumerate(["Balanced", "Skewed"]):
            ax = axes[row_idx, col_idx]
            seq_rows = filter_rows(rows, dsize, profile, "Sequential")
            if seq_rows:
                seq_ms = seq_rows[0]["median_total_ms"]
                ax.axhline(
                    seq_ms,
                    color=POLICY_STYLES["Sequential"]["color"],
                    linestyle=POLICY_STYLES["Sequential"]["linestyle"],
                    linewidth=1.5,
                    label=POLICY_STYLES["Sequential"]["label"] if (row_idx == 0 and col_idx == 0) else None,
                )

            for policy in ["Static", "Dynamic", "Guided", "Adaptive"]:
                p_rows = filter_rows(rows, dsize, profile, policy)
                if not p_rows:
                    continue
                threads = [r["configured_threads"] for r in p_rows]
                medians = [r["median_total_ms"] for r in p_rows]
                iqrs = [0.5 * r["iqr_total_ms"] for r in p_rows]
                st = POLICY_STYLES[policy]
                ax.errorbar(
                    threads,
                    medians,
                    yerr=iqrs,
                    color=st["color"],
                    marker=st["marker"],
                    linestyle=st["linestyle"],
                    linewidth=1.8,
                    markersize=6,
                    capsize=3,
                    label=st["label"] if (row_idx == 0 and col_idx == 0) else None,
                )

            ax.set_title(f"N = {dsize} Documents ({profile} Profile)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Median Runtime (ms)")
            ax.set_xticks([1, 2, 4, 8])
            ax.grid(True, linestyle=":", alpha=0.6)

    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel("Configured OpenMP Threads (p)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("End-to-End Pipeline Runtime vs. Thread Count (±0.5 IQR)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "runtime_vs_threads.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_speedup_vs_threads(rows, out_dir):
    sizes = sorted(list({r["dataset_size"] for r in rows}))
    target_sizes = [s for s in [1000, 2000, 4000, 8000] if s in sizes]
    if not target_sizes:
        target_sizes = sizes

    fig, axes = plt.subplots(len(target_sizes), 2, figsize=(13, 3.3 * len(target_sizes)), sharex=True)
    if len(target_sizes) == 1:
        axes = np.array([axes])

    for row_idx, dsize in enumerate(target_sizes):
        for col_idx, profile in enumerate(["Balanced", "Skewed"]):
            ax = axes[row_idx, col_idx]
            ax.plot(
                [1, 2, 4, 8],
                [1.0, 1.0, 1.0, 1.0],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="Sequential Baseline (1.0x)" if (row_idx == 0 and col_idx == 0) else None,
            )

            for policy in ["Static", "Dynamic", "Guided", "Adaptive"]:
                p_rows = filter_rows(rows, dsize, profile, policy)
                if not p_rows:
                    continue
                threads = [r["configured_threads"] for r in p_rows]
                speedups = [r["speedup_total"] for r in p_rows]
                st = POLICY_STYLES[policy]
                ax.plot(
                    threads,
                    speedups,
                    color=st["color"],
                    marker=st["marker"],
                    linestyle=st["linestyle"],
                    linewidth=1.8,
                    markersize=6,
                    label=st["label"] if (row_idx == 0 and col_idx == 0) else None,
                )

            ax.set_title(f"N = {dsize} Documents ({profile} Profile)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Speedup S_p = T_seq / T_p")
            ax.set_xticks([1, 2, 4, 8])
            ax.grid(True, linestyle=":", alpha=0.6)

    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel("Configured OpenMP Threads (p)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Parallel Speedup vs. Thread Count Across Dataset Sizes", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "speedup_vs_threads.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_efficiency_vs_threads(rows, out_dir):
    sizes = sorted(list({r["dataset_size"] for r in rows}))
    target_sizes = [s for s in [1000, 2000, 4000, 8000] if s in sizes]
    if not target_sizes:
        target_sizes = sizes

    fig, axes = plt.subplots(len(target_sizes), 2, figsize=(13, 3.3 * len(target_sizes)), sharex=True)
    if len(target_sizes) == 1:
        axes = np.array([axes])

    for row_idx, dsize in enumerate(target_sizes):
        for col_idx, profile in enumerate(["Balanced", "Skewed"]):
            ax = axes[row_idx, col_idx]
            ax.axhline(
                1.0,
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="Ideal Linear Efficiency (1.0)" if (row_idx == 0 and col_idx == 0) else None,
            )

            for policy in ["Static", "Dynamic", "Guided", "Adaptive"]:
                p_rows = filter_rows(rows, dsize, profile, policy)
                if not p_rows:
                    continue
                threads = [r["configured_threads"] for r in p_rows]
                effs = [r["efficiency_total"] for r in p_rows]
                st = POLICY_STYLES[policy]
                ax.plot(
                    threads,
                    effs,
                    color=st["color"],
                    marker=st["marker"],
                    linestyle=st["linestyle"],
                    linewidth=1.8,
                    markersize=6,
                    label=st["label"] if (row_idx == 0 and col_idx == 0) else None,
                )

            ax.set_title(f"N = {dsize} Documents ({profile} Profile)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Efficiency E_p = S_p / p")
            ax.set_xticks([1, 2, 4, 8])
            ax.set_ylim(0.0, 1.25)
            ax.grid(True, linestyle=":", alpha=0.6)

    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel("Configured OpenMP Threads (p)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Parallel Efficiency vs. Thread Count Across Dataset Sizes", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "efficiency_vs_threads.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_work_imbalance_and_stages(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left panel: Estimated Pair-Work Imbalance Ratio I vs Threads for N=2000 & N=8000 Skewed
    ax_imb = axes[0]
    target_dsize = 8000 if any(r["dataset_size"] == 8000 for r in rows) else max(r["dataset_size"] for r in rows)
    ax_imb.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, label="Ideal Balance (I = 1.0)")

    for policy in ["Static", "Dynamic", "Guided", "Adaptive"]:
        p_rows = filter_rows(rows, target_dsize, "Skewed", policy)
        if not p_rows:
            continue
        threads = [r["configured_threads"] for r in p_rows]
        imbalances = [r["work_imbalance_ratio"] for r in p_rows]
        st = POLICY_STYLES[policy]
        ax_imb.plot(
            threads,
            imbalances,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=2.0,
            markersize=7,
            label=st["label"],
        )

    ax_imb.set_title(f"Estimated Pair-Work Imbalance Ratio I (N={target_dsize}, Skewed)", fontsize=11, fontweight="bold")
    ax_imb.set_xlabel("Configured OpenMP Threads (p)")
    ax_imb.set_ylabel("Work Imbalance I = max_t(W_t) / mean_t(W_t)")
    ax_imb.set_xticks([1, 2, 4, 8])
    ax_imb.legend(loc="upper left", fontsize=9)
    ax_imb.grid(True, linestyle=":", alpha=0.6)

    # Right panel: Stage Breakdown at N=target_dsize Skewed (T=1, 2, 4, 8 for Adaptive vs Fixed T=4/8)
    ax_stg = axes[1]
    adap_rows = filter_rows(rows, target_dsize, "Skewed", "Adaptive")
    static_rows = filter_rows(rows, target_dsize, "Skewed", "Static")
    seq_rows = filter_rows(rows, target_dsize, "Skewed", "Sequential")

    labels = []
    fp_vals = []
    bkt_vals = []
    pair_gen_vals = []
    merge_dedup_vals = []
    ver_vals = []

    if seq_rows:
        r = seq_rows[0]
        labels.append("Seq\n(T=1)")
        fp_vals.append(r["median_fingerprinting_ms"])
        bkt_vals.append(r["median_bucket_construction_ms"])
        pair_gen_vals.append(r["median_candidate_gen_total_ms"])
        merge_dedup_vals.append(0.0)
        ver_vals.append(r["median_verification_ms"])

    for r in static_rows:
        if r["configured_threads"] in (4, 8):
            labels.append(f"Static\n(T={r['configured_threads']})")
            fp_vals.append(r["median_fingerprinting_ms"])
            bkt_vals.append(r["median_bucket_construction_ms"])
            pair_gen_vals.append(r["median_candidate_gen_total_ms"])
            merge_dedup_vals.append(0.0)
            ver_vals.append(r["median_verification_ms"])

    for r in adap_rows:
        t = r["configured_threads"]
        labels.append(f"Adaptive\n(T={t})")
        fp_vals.append(r["median_fingerprinting_ms"])
        bkt_vals.append(r["median_bucket_construction_ms"])
        # Parallel pair generation = decision + ordinary + heavy chunk gen + heavy pair proc
        pgen = (
            r["median_adaptive_decision_ms"]
            + r["median_ordinary_candidate_gen_ms"]
            + r["median_heavy_chunk_gen_ms"]
            + r["median_heavy_pair_proc_ms"]
        )
        pair_gen_vals.append(pgen)
        merge_dedup_vals.append(r["median_merge_sort_dedup_ms"])
        ver_vals.append(r["median_verification_ms"])

    x = np.arange(len(labels))
    fp_arr = np.array(fp_vals)
    bkt_arr = np.array(bkt_vals)
    pgen_arr = np.array(pair_gen_vals)
    mrg_arr = np.array(merge_dedup_vals)
    ver_arr = np.array(ver_vals)

    ax_stg.bar(x, fp_arr, label="1. Fingerprinting", color="#4c78a8")
    ax_stg.bar(x, bkt_arr, bottom=fp_arr, label="2. Bucket Construction", color="#72b7b2")
    ax_stg.bar(
        x,
        pgen_arr,
        bottom=fp_arr + bkt_arr,
        label="3a-d. CandGen (Full Stage 3 for Seq/Static)",
        color="#f58518",
    )
    ax_stg.bar(
        x,
        mrg_arr,
        bottom=fp_arr + bkt_arr + pgen_arr,
        label="3e. Sequential Merge/Sort/Dedup (Adaptive)",
        color="#e45756",
    )
    ax_stg.bar(
        x,
        ver_arr,
        bottom=fp_arr + bkt_arr + pgen_arr + mrg_arr,
        label="4. Hamming Verification",
        color="#54a24b",
    )

    ax_stg.set_title(f"Pipeline Stage Breakdown (N={target_dsize}, Skewed)", fontsize=11, fontweight="bold")
    ax_stg.set_ylabel("Median Stage Runtime (ms)")
    ax_stg.set_xticks(x)
    ax_stg.set_xticklabels(labels, fontsize=9)
    ax_stg.legend(loc="upper right", fontsize=8.5)
    ax_stg.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "work_imbalance_and_stages.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 6 benchmark plots from summary CSV.")
    parser.add_argument("--csv", default="results/benchmark_summary.csv", help="Path to summary CSV")
    parser.add_argument("--out-dir", default="results", help="Output directory for generated plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_summary_csv(args.csv)

    p1 = plot_runtime_vs_threads(rows, args.out_dir)
    p2 = plot_speedup_vs_threads(rows, args.out_dir)
    p3 = plot_efficiency_vs_threads(rows, args.out_dir)
    p4 = plot_work_imbalance_and_stages(rows, args.out_dir)

    print("Generated plots:")
    for p in (p1, p2, p3, p4):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
