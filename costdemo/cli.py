#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import APIError
from tabulate import tabulate

from costdemo.costing import estimate_cost
from costdemo.foundry_client import create_client, run_chat_completion
from costdemo.models import UsageNumbers
from costdemo.utils import float_from_env, obj_to_dict

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

logger = logging.getLogger(__name__)


def int_or_none(value: object | None) -> int | None:
    """Convert common numeric-like values to int when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def int_or_default(value: object | None, default: int = 0) -> int:
    """Convert value to int or return a default fallback."""
    converted = int_or_none(value)
    if converted is None:
        return default
    return converted


def extract_usage(response: object) -> tuple[UsageNumbers, dict[str, object]]:
    """Extract usage counters from SDK response model."""
    usage = getattr(response, "usage", None)
    usage_dict = obj_to_dict(usage)
    return (
        UsageNumbers(
            prompt_tokens=int_or_none(usage_dict.get("prompt_tokens")),
            completion_tokens=int_or_none(usage_dict.get("completion_tokens")),
            total_tokens=int_or_none(usage_dict.get("total_tokens")),
        ),
        usage_dict,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the cost summary demo command."""
    parser = argparse.ArgumentParser(
        description="Run a Foundry chat call and print full token usage details."
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Foundry endpoint, for example https://<account>.cognitiveservices.azure.com/",
    )
    parser.add_argument("--api-key", default=None, help="API key for the Foundry account.")
    parser.add_argument(
        "--api-version",
        default="2024-10-21",
        help="Azure OpenAI API version.",
    )
    parser.add_argument(
        "--deployment",
        default="cost-helper-chat",
        help="Model deployment name in your Foundry account.",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize cloud costs for app + database + monitoring with a concise recommendation.",
        help="User prompt sent to the model.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Optional USD input token price per 1M tokens for cost estimation.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Optional USD output token price per 1M tokens for cost estimation.",
    )
    parser.add_argument(
        "--cached-input-price-per-1m",
        type=float,
        default=None,
        help="Optional USD cached input token price per 1M tokens for cost estimation.",
    )
    parser.add_argument(
        "--show-raw-usage",
        action="store_true",
        help="Print the full raw usage payload as JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def configure_logging(verbose: bool = False) -> None:
    """Configure root logger for command output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def run(args: argparse.Namespace) -> int:
    """Execute one Foundry request and print token/cost details."""
    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    deployment = args.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = args.api_version or os.getenv("AZURE_OPENAI_API_VERSION")

    input_price_per_1m = (
        args.input_price_per_1m
        if args.input_price_per_1m is not None
        else float_from_env("INPUT_PRICE_PER_1M", "AZURE_OPENAI_INPUT_PRICE_PER_1M")
    )
    output_price_per_1m = (
        args.output_price_per_1m
        if args.output_price_per_1m is not None
        else float_from_env("OUTPUT_PRICE_PER_1M", "AZURE_OPENAI_OUTPUT_PRICE_PER_1M")
    )
    cached_input_price_per_1m = (
        args.cached_input_price_per_1m
        if args.cached_input_price_per_1m is not None
        else float_from_env(
            "CACHED_INPUT_PRICE_PER_1M", "AZURE_OPENAI_CACHED_INPUT_PRICE_PER_1M"
        )
    )

    if not endpoint:
        logger.error("Missing endpoint. Set --endpoint or AZURE_OPENAI_ENDPOINT.")
        return EXIT_ERROR
    if not api_key:
        logger.error("Missing API key. Set --api-key or AZURE_OPENAI_API_KEY.")
        return EXIT_ERROR
    if not deployment:
        logger.error("Missing deployment. Set --deployment or AZURE_OPENAI_DEPLOYMENT.")
        return EXIT_ERROR
    if not api_version:
        logger.error("Missing API version. Set --api-version or AZURE_OPENAI_API_VERSION.")
        return EXIT_ERROR

    logger.debug("Endpoint: %s", endpoint)
    logger.debug("Deployment: %s", deployment)

    client = create_client(
        endpoint=endpoint,
        deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )

    try:
        response = run_chat_completion(client, deployment=deployment, prompt=args.prompt)
    except APIError as exc:
        logger.error("Azure OpenAI request failed: %s", exc)
        return EXIT_FAILURE

    content = ""
    response_obj = response if isinstance(response, dict) else getattr(response, "__dict__", {})
    choices: Any = getattr(response, "choices", None)
    if choices is None and isinstance(response_obj, dict):
        choices = response_obj.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")
        content = getattr(message, "content", "") or ""
        if not content and isinstance(message, dict):
            content = str(message.get("content") or "")

    usage_numbers, usage_raw = extract_usage(response)
    prompt_tokens_details = obj_to_dict(usage_raw.get("prompt_tokens_details"))
    cached_prompt_tokens = int_or_default(prompt_tokens_details.get("cached_tokens"), 0)

    print("\n=== Assistant Response ===")
    print(content.strip() or "<empty response>")

    usage_rows = [
        ["prompt_tokens", usage_numbers.prompt_tokens],
        ["completion_tokens", usage_numbers.completion_tokens],
        ["total_tokens", usage_numbers.total_tokens],
    ]
    print("\n=== Token Usage ===")
    print(tabulate(usage_rows, headers=["metric", "value"], tablefmt="github"))

    costs = estimate_cost(
        usage=usage_numbers,
        cached_prompt_tokens=cached_prompt_tokens,
        input_price_per_1m=input_price_per_1m,
        cached_input_price_per_1m=cached_input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
    )
    if costs:
        cost_rows = [
            ["prompt_tokens_non_cached", f"{costs['prompt_tokens_non_cached']:.0f}"],
            ["prompt_tokens_cached", f"{costs['prompt_tokens_cached']:.0f}"],
            ["input_cost_non_cached_usd", f"{costs['input_cost_non_cached_usd']:.8f}"],
            ["input_cost_cached_usd", f"{costs['input_cost_cached_usd']:.8f}"],
            ["input_cost_usd", f"{costs['input_cost_usd']:.8f}"],
            ["output_cost_usd", f"{costs['output_cost_usd']:.8f}"],
            ["total_cost_usd", f"{costs['total_cost_usd']:.8f}"],
        ]
        print("\n=== Estimated Cost (USD) ===")
        print(tabulate(cost_rows, headers=["metric", "value"], tablefmt="github"))

    if args.show_raw_usage:
        print("\n=== Raw Usage Payload ===")
        print(json.dumps(usage_raw, indent=2, sort_keys=True))

    response_id = getattr(response, "id", None)
    if response_id:
        print(f"\nresponse_id: {response_id}")

    return EXIT_SUCCESS


def main() -> int:
    """CLI entrypoint with robust error handling."""
    load_dotenv()
    args = create_parser().parse_args()
    configure_logging(verbose=args.verbose)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except BrokenPipeError:
        sys.stderr.close()
        return EXIT_FAILURE
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
