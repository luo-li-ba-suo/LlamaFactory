#!/usr/bin/env python3
"""Prepare BT eval dataset from HelpSteer2 val + RM-Bench data, with tags (sharegpt format)."""

import json
import random
from collections import defaultdict
from pathlib import Path

import tiktoken


random.seed(42)

HELPSTEER2_BT_VAL = "/workspace/projects/reward_bench/data/helpsteer2/assembled/helpsteer2/bt_val.json"
RM_BENCH_TOTAL = "/workspace/projects/reward_bench/data/rm-bench/total_dataset.json"
OUTPUT = "/workspace/projects/reward_bench/data/helpsteer2/final/bt_eval_tagged.json"

DOMAIN_TAG_MAP = {
    "chat": "chat",
    "code": "code",
    "math": "math",
    "safety-refuse": "safety",
    "safety-response": "safety",
}

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


def load_helpsteer2_bt():
    with open(HELPSTEER2_BT_VAL) as f:
        data = json.load(f)
    for item in data:
        item["tags"] = ["helpsteer2"]
    return data


def load_rm_bench_bt():
    with open(RM_BENCH_TOTAL) as f:
        raw = json.load(f)

    by_diff_domain = defaultdict(lambda: defaultdict(list))

    for unit in raw:
        domain_tag = DOMAIN_TAG_MAP.get(unit["domain"], unit["domain"])
        for i in range(3):
            for j in range(3):
                diff = DIFFICULTY[(i, j)]
                # Extract the last assistant reply as the chosen/rejected text
                # RM-Bench data assumes chosen[i] and rejected[j] are full assistant replies
                by_diff_domain[diff][domain_tag].append(
                    {
                        "conversations": [{"from": "human", "value": unit["prompt"]}],
                        "chosen": {"from": "gpt", "value": unit["chosen"][i]},
                        "rejected": {"from": "gpt", "value": unit["rejected"][j]},
                    }
                )

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
    hs2 = load_helpsteer2_bt()
    rmb = load_rm_bench_bt()

    MAX_HS2 = 500
    if len(hs2) > MAX_HS2:
        hs2 = random.sample(hs2, MAX_HS2)

    combined = hs2 + rmb
    random.shuffle(combined)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # Stats
    print(f"\nHelpSteer2 val: {len(hs2)} samples")
    print(f"RM-Bench:       {len(rmb)} samples")
    print(f"Combined:       {len(combined)} samples")
    print(f"Output:         {OUTPUT}")

    tag_counts = {}
    for item in combined:
        for tag in item.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\nTag distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    # Token length stats
    enc = tiktoken.get_encoding("cl100k_base")
    chosen_by_tag = defaultdict(list)
    rejected_by_tag = defaultdict(list)
    for item in combined:
        conv_text = "".join(m["value"] for m in item.get("conversations", []))
        chosen_text = conv_text + item["chosen"]["value"]
        rejected_text = conv_text + item["rejected"]["value"]
        c_len = len(enc.encode(chosen_text))
        r_len = len(enc.encode(rejected_text))
        for tag in item.get("tags", []):
            chosen_by_tag[tag].append(c_len)
            rejected_by_tag[tag].append(r_len)
    print_length_stats(chosen_by_tag, "chosen (prompt+chosen)")
    print_length_stats(rejected_by_tag, "rejected (prompt+rejected)")


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
