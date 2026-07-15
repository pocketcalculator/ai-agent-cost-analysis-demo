#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate monthly token costs for a projected request volume."
    )
    parser.add_argument("--requests-per-month", type=int, required=True)
    parser.add_argument("--avg-prompt-tokens", type=int, required=True)
    parser.add_argument("--avg-completion-tokens", type=int, required=True)
    parser.add_argument("--input-price-per-1m", type=float, required=True)
    parser.add_argument("--output-price-per-1m", type=float, required=True)
    return parser


def main() -> int:
    args = create_parser().parse_args()

    monthly_prompt_tokens = args.requests_per_month * args.avg_prompt_tokens
    monthly_completion_tokens = args.requests_per_month * args.avg_completion_tokens

    monthly_input_cost = (monthly_prompt_tokens / 1_000_000) * args.input_price_per_1m
    monthly_output_cost = (monthly_completion_tokens / 1_000_000) * args.output_price_per_1m
    monthly_total_cost = monthly_input_cost + monthly_output_cost

    print("monthly_prompt_tokens:", monthly_prompt_tokens)
    print("monthly_completion_tokens:", monthly_completion_tokens)
    print("monthly_input_cost_usd:", f"{monthly_input_cost:.2f}")
    print("monthly_output_cost_usd:", f"{monthly_output_cost:.2f}")
    print("monthly_total_cost_usd:", f"{monthly_total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
