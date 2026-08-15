#!/usr/bin/env python3
"""Token length distribution analysis for all datasets (train + eval, BT + SfL)."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tiktoken


matplotlib.use("Agg")

ENC = tiktoken.get_encoding("cl100k_base")
BASE = Path("/workspace/projects/reward_bench/data/helpsteer2")
OUT = BASE / "token_length_plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── Dataset registry ────────────────────────────────────────────────────────
DATASETS = {
    "BT Train": (BASE / "assembled" / "helpsteer2" / "bt_train.json", "bt"),
    "BT Eval": (BASE / "final" / "bt_eval_tagged.json", "bt"),
    "SfL 1-5 Train": (BASE / "assembled" / "helpsteer2" / "sfl_1_5_train.json", "sfl"),
    "SfL 1-5 Eval": (BASE / "final" / "sfl_1_5_eval_tagged.json", "sfl"),
    "SfL bin Train": (BASE / "assembled" / "helpsteer2" / "sfl_binary_train.json", "sfl"),
    "SfL bin Eval": (BASE / "final" / "sfl_binary_eval_tagged.json", "sfl"),
}

TAG_ORDER = ["hard", "normal", "easy", "chat", "code", "math", "safety", "helpsteer2"]


# ── Helpers ─────────────────────────────────────────────────────────────────


def tokenize(text: str) -> int:
    return len(ENC.encode(text))


def load_and_compute(path: Path, fmt: str):
    """Load dataset, return lengths and per-tag stats."""
    with open(path) as f:
        data = json.load(f)

    all_chosen, all_rejected = [], []
    by_tag_c = defaultdict(list)
    by_tag_r = defaultdict(list)

    for item in data:
        if fmt == "bt":
            conv_text = "".join(m["value"] for m in item.get("conversations", []))
            c_text = conv_text + item["chosen"]["value"]
            r_text = conv_text + item["rejected"]["value"]
        else:  # sfl messages format
            c_text = item["chosen"][0]["value"]
            r_text = item["rejected"][0]["value"]

        c_len = tokenize(c_text)
        r_len = tokenize(r_text)
        all_chosen.append(c_len)
        all_rejected.append(r_len)

        tags = item.get("tags")
        if tags:
            for tag in tags:
                by_tag_c[tag].append(c_len)
                by_tag_r[tag].append(r_len)
        else:
            by_tag_c["_overall"].append(c_len)
            by_tag_r["_overall"].append(r_len)

    return all_chosen, all_rejected, by_tag_c, by_tag_r


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_histogram(ax, chosen, rejected, title, bins=80):
    colors = {"chosen": "#2196F3", "rejected": "#FF5722"}
    for arr, label in [(chosen, "chosen"), (rejected, "rejected")]:
        ax.hist(arr, bins=bins, alpha=0.55, color=colors[label], label=label, edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Tokens (cl100k_base)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def draw_tag_boxplot(ax, tags, by_tag_c, by_tag_r, title):
    """Simple side-by-side boxplot for chosen vs rejected per tag."""
    n = len(tags)
    xs = np.arange(n)
    bp_c = ax.boxplot([by_tag_c[t] for t in tags], positions=xs - 0.22, widths=0.35, patch_artist=True)
    bp_r = ax.boxplot([by_tag_r[t] for t in tags], positions=xs + 0.22, widths=0.35, patch_artist=True)
    for p in bp_c["boxes"]:
        p.set_facecolor("#2196F3")
        p.set_alpha(0.7)
    for p in bp_r["boxes"]:
        p.set_facecolor("#FF5722")
        p.set_alpha(0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(tags, fontsize=8, rotation=30 if n > 5 else 0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Tokens")
    ax.legend([bp_c["boxes"][0], bp_r["boxes"][0]], ["chosen", "rejected"], fontsize=7)
    ax.grid(axis="y", alpha=0.3)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("Loading datasets...")
    all_data = {}
    for name, (path, fmt) in DATASETS.items():
        chosen, rejected, by_tag_c, by_tag_r = load_and_compute(path, fmt)
        all_data[name] = (chosen, rejected, by_tag_c, by_tag_r)
        print(
            f"  {name}: {len(chosen)} samples, "
            f"chosen mean={np.mean(chosen):.0f}, rejected mean={np.mean(rejected):.0f}"
        )

    # ── Max token summary ──
    print(f"\n{'=' * 70}")
    print("  MAX TOKEN LENGTHS (for cutoff reference)")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<18} {'chosen':>10} {'rejected':>10} {'overall':>10}")
    print(f"{'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, (c, r, _, _) in all_data.items():
        print(f"{name:<18} {max(c):>10} {max(r):>10} {max(max(c), max(r)):>10}")

    # ── Figure 1: Overall histograms ──
    n = len(all_data)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, (name, (c, r, _, _)) in zip(axes_flat, all_data.items()):
        plot_histogram(ax, c, r, name)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Token Length Distributions — Overall", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "overall_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Figure 2: Per-dataset tag boxplots (eval sets only) ──
    for name, (_, _, by_tag_c, by_tag_r) in all_data.items():
        tags_present = [t for t in TAG_ORDER if t in by_tag_c and t != "_overall"]
        if not tags_present:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(tags_present) * 0.7 + 1), 5))
        draw_tag_boxplot(ax, tags_present, by_tag_c, by_tag_r, f"{name} — Token Length by Tag")
        fig.tight_layout()
        safe = name.lower().replace(" ", "_")
        path = OUT / f"{safe}_boxplots.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Console stats ──
    for name, (c, r, by_tag_c, by_tag_r) in all_data.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}  ({len(c)} samples)")
        print(f"{'=' * 70}")
        print_token_stats(by_tag_c, by_tag_r)

    print(f"\nPlots saved to: {OUT}")


def print_token_stats(by_tag_c, by_tag_r):
    all_tags = set(by_tag_c.keys()) | set(by_tag_r.keys())

    def fmt(a):
        s = sorted(a)
        n = len(s)
        return (
            sum(s) // n,
            s[0],
            s[-1],
            s[n // 4],
            s[n // 2],
            s[3 * n // 4],
            s[9 * n // 10],
            s[max(0, 99 * n // 100 - 1)],
        )

    ordered = [t for t in TAG_ORDER if t in all_tags]
    ordered += sorted(all_tags - set(TAG_ORDER) - {"_overall"})
    if "_overall" in all_tags:
        ordered.append("(untagged)")

    hdr = f"{'Tag':<22} {'Field':>8} {'Mean':>7} {'Min':>6} {'Max':>6} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'P99':>6}"
    sep = f"{'-' * 22} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}"
    print(hdr)
    print(sep)
    for tag_disp in ordered:
        tag = "_overall" if tag_disp == "(untagged)" else tag_disp
        for field, m in [("chosen", by_tag_c), ("rejected", by_tag_r)]:
            if tag not in m:
                continue
            mean, mn, mx, p25, p50, p75, p90, p99 = fmt(m[tag])
            print(f"{tag_disp:<22} {field:>8} {mean:>7} {mn:>6} {mx:>6} {p25:>6} {p50:>6} {p75:>6} {p90:>6} {p99:>6}")
        print()


if __name__ == "__main__":
    main()
