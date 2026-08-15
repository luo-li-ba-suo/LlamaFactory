#!/usr/bin/env python3
"""Extract the canonical 600-row RM-Bench split from the Plan C eval set."""

import argparse
import json
from collections import Counter
from pathlib import Path


DOMAINS = ("chat", "code", "math", "safety")
DIFFICULTIES = ("easy", "normal", "hard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/workspace/projects/reward_bench/data/helpsteer3/curriculum/qwen3_8b_sfl_plan_c_round1_v1/eval.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/projects/reward_bench/data/helpsteer3/final/rm_bench_eval_600.json"),
    )
    args = parser.parse_args()

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    rm_rows = [row for row in rows if len(row.get("tags", [])) == 2 and row["tags"][0] in DOMAINS]
    cells = Counter(tuple(row["tags"]) for row in rm_rows)
    expected = {(domain, difficulty): 50 for domain in DOMAINS for difficulty in DIFFICULTIES}
    if len(rm_rows) != 600 or cells != expected:
        raise ValueError(f"expected 600 rows with 50 per domain/difficulty cell, got {len(rm_rows)} and {cells}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rm_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rm_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
