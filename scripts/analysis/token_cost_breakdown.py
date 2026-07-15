#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a saved usage JSON payload and print token totals."
    )
    parser.add_argument("usage_json", help="Path to a usage payload JSON file")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    with open(args.usage_json, "r", encoding="utf-8") as f:
        usage = json.load(f)

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    print("prompt_tokens:", prompt_tokens)
    print("completion_tokens:", completion_tokens)
    print("total_tokens:", total_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
