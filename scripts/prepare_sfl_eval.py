#!/usr/bin/env python3
"""Prepare SfL eval datasets: read assembled data, add tags, sample, merge.

Reads assembled helpsteer2 val + rm-bench data for each template, applies tags
(difficulty + domain for RM-Bench, "helpsteer2" for HelpSteer2), samples evenly,
and writes final eval sets.

Usage:
    python scripts/prepare_sfl_eval.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import tiktoken


random.seed(42)

BASE = Path("/workspace/projects/reward_bench/data/helpsteer2")
ASSEMBLED_HS2 = BASE / "assembled" / "helpsteer2"
ASSEMBLED_RMB = BASE / "assembled" / "rm-bench"
FINAL = BASE / "final"

TEMPLATES = ["1_5", "binary"]
MAX_HELPSTEER2 = 500

DIFFICULTY = {
    (1, 0): "easy",
    (2, 0): "easy",
    (2, 1): "easy",
    (0, 0): "normal",
    (1, 1): "normal",
    (2, 2): "normal",
    (0, 1): "hard",
    (0, 2): "hard",
    (1, 2): "hard",
}

DOMAIN_TAG_MAP = {
    "chat": "chat",
    "code": "code",
    "math": "math",
    "safety-refuse": "safety",
    "safety-response": "safety",
}


def load_helpsteer2_val(tmpl: str) -> list[dict]:
    path = ASSEMBLED_HS2 / f"sfl_{tmpl}_val.json"
    with open(path) as f:
        data = json.load(f)
    for item in data:
        item["tags"] = ["helpsteer2"]
    return data


def load_rm_bench(tmpl: str) -> list[dict]:
    """Load assembled RM-Bench data, add difficulty + domain tags, sample evenly."""
    path = ASSEMBLED_RMB / f"sfl_{tmpl}.json"
    with open(path) as f:
        assembled = json.load(f)

    # Load raw RM-Bench for domain lookup by sample_id
    rmb_raw_path = Path("/workspace/projects/reward_bench/data/rm-bench/total_dataset.json")
    with open(rmb_raw_path) as f:
        raw = json.load(f)
    id_to_domain = {unit["id"]: DOMAIN_TAG_MAP.get(unit["domain"], unit["domain"]) for unit in raw}

    # Group by difficulty × domain
    by_diff_domain = defaultdict(lambda: defaultdict(list))
    for item in assembled:
        i, j = map(int, item["pair"].split("_"))
        diff = DIFFICULTY[(i, j)]
        domain = id_to_domain.get(item["sample_id"], "unknown")
        by_diff_domain[diff][domain].append(item)

    # Sample evenly
    PER_DIFF_PER_DOMAIN = 50
    output = []
    for domain_tag in ["chat", "code", "math", "safety"]:
        for diff in ["hard", "normal", "easy"]:
            items = by_diff_domain[diff].get(domain_tag, [])
            available = min(len(items), PER_DIFF_PER_DOMAIN)
            print(f"  {domain_tag}/{diff}: {len(items)} available, sampling {available}")
            random.shuffle(items)
            sampled = items[:available]
            for item in sampled:
                item["tags"] = [domain_tag, diff]
            output.extend(sampled)
    return output


def main():
    enc = tiktoken.get_encoding("cl100k_base")
    FINAL.mkdir(parents=True, exist_ok=True)

    for tmpl in TEMPLATES:
        print(f"\n{'=' * 60}")
        print(f"  Template: {tmpl}")
        print(f"{'=' * 60}")

        hs2 = load_helpsteer2_val(tmpl)
        rmb = load_rm_bench(tmpl)

        if len(hs2) > MAX_HELPSTEER2:
            hs2 = random.sample(hs2, MAX_HELPSTEER2)

        combined = hs2 + rmb
        random.shuffle(combined)

        out_path = FINAL / f"sfl_{tmpl}_eval_tagged.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\nHelpSteer2 val: {len(hs2)} samples")
        print(f"RM-Bench:       {len(rmb)} samples")
        print(f"Combined:       {len(combined)} samples")
        print(f"Output:         {out_path}")

        # Tag distribution
        tag_counts: dict[str, int] = {}
        for item in combined:
            for tag in item.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        print("\nTag distribution:")
        for tag, count in sorted(tag_counts.items()):
            print(f"  {tag}: {count}")

        # Token length stats
        chosen_by_tag = defaultdict(list)
        rejected_by_tag = defaultdict(list)
        for item in combined:
            c_len = len(enc.encode(item["chosen"][0]["value"]))
            r_len = len(enc.encode(item["rejected"][0]["value"]))
            for tag in item.get("tags", []):
                chosen_by_tag[tag].append(c_len)
                rejected_by_tag[tag].append(r_len)
        print_length_stats(chosen_by_tag, f"chosen [{tmpl}]")
        print_length_stats(rejected_by_tag, f"rejected [{tmpl}]")


def print_length_stats(lens_by_tag, label):
    """Print distribution stats for a set of token-length lists keyed by tag."""
    print(f"\n{'=' * 60}")
    print(f"Token length distribution — {label}")
    print(f"{'=' * 60}")
    print(f"{'Tag':<25} {'Mean':>7} {'Min':>6} {'Max':>6} {'P25':>6} {'P50':>6} {'P75':>6} {'P90':>6} {'P99':>6}")
    print(f"{'-' * 25} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")
    for tag in sorted(lens_by_tag.keys()):
        arr = sorted(lens_by_tag[tag])
        n = len(arr)
        print(
            f"{tag:<25} {sum(arr) // n:>7} {arr[0]:>6} {arr[-1]:>6} "
            f"{arr[n // 4]:>6} {arr[n // 2]:>6} {arr[3 * n // 4]:>6} {arr[9 * n // 10]:>6} {arr[max(0, 99 * n // 100 - 1)]:>6}"
        )


if __name__ == "__main__":
    main()
