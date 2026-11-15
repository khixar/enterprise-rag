#!/usr/bin/env python3
"""
Total the cost of the API's external model calls from its USAGE log lines.

Prices are list prices in USD and may be out of date; check the providers' pricing pages.

Usage:
    python scripts/cost_report.py --since 2026-11-15T10:00:00Z [--until 2026-11-15T10:30:00Z]
"""
import argparse
from collections import defaultdict
import json
import subprocess

# model -> (USD per 1M input tokens, USD per 1M output tokens, USD per search unit)
PRICES = {
    "gpt-4o-mini": (0.15, 0.60, 0.0),
    "gpt-4o": (2.50, 10.00, 0.0),
    "text-embedding-3-small": (0.02, 0.0, 0.0),
    "rerank-english-v3.0": (0.0, 0.0, 2.00 / 1000),
}


def cost_of(model: str, input_tokens: int, output_tokens: int, search_units: int = 0) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_out, p_search = PRICES[model]
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out + search_units * p_search


def print_table(totals: dict[str, dict[str, int]]) -> float:
    grand = 0.0
    print(f"  {'model':26}{'calls':>7}{'input tok':>12}{'output tok':>12}{'searches':>10}{'cost $':>10}")
    for model, t in sorted(totals.items()):
        c = cost_of(model, t["input_tokens"], t["output_tokens"], t["search_units"])
        grand += c or 0.0
        cost = f"{c:.4f}" if c is not None else "unknown"
        print(f"  {model:26}{t['calls']:>7}{t['input_tokens']:>12,}{t['output_tokens']:>12,}{t['search_units']:>10}{cost:>10}")
    print(f"  {'TOTAL':26}{'':>41}{grand:>10.4f}")
    return grand


def main():
    parser = argparse.ArgumentParser(description="Sum API model-call costs from docker logs")
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", default=None)
    args = parser.parse_args()

    cmd = ["docker", "compose", "logs", "api", "--no-log-prefix", "--since", args.since]
    if args.until:
        cmd += ["--until", args.until]
    logs = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in logs.splitlines():
        if not line.startswith("USAGE "):
            continue
        u = json.loads(line[len("USAGE "):])
        t = totals[u["model"]]
        t["calls"] += 1
        for k in ("input_tokens", "output_tokens", "search_units"):
            t[k] += u[k]
    print_table(totals)


if __name__ == "__main__":
    main()
