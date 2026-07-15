#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate request cost from a usage JSON payload and token prices."
    )
    parser.add_argument("usage_json", help="Path to a usage payload JSON file")
    parser.add_argument("--input-price-per-1m", type=float, required=True)
    parser.add_argument("--output-price-per-1m", type=float, required=True)
    parser.add_argument("--cached-input-price-per-1m", type=float, default=None)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    with open(args.usage_json, "r", encoding="utf-8") as f:
        usage = json.load(f)

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    cached_tokens = int(
        ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    )

    non_cached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
    cached_price = (
        args.cached_input_price_per_1m
        if args.cached_input_price_per_1m is not None
        else args.input_price_per_1m
    )

    input_cost = (
        (non_cached_prompt_tokens / 1_000_000) * args.input_price_per_1m
        + (cached_tokens / 1_000_000) * cached_price
    )
    output_cost = (completion_tokens / 1_000_000) * args.output_price_per_1m
    total_cost = input_cost + output_cost

    print("input_cost_usd:", f"{input_cost:.8f}")
    print("output_cost_usd:", f"{output_cost:.8f}")
    print("total_cost_usd:", f"{total_cost:.8f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
