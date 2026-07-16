#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, cast

from dotenv import load_dotenv
from openai import APIError
from tabulate import tabulate

# Ensure the repository root is importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from costdemo.foundry_client import create_client
from costdemo.costing import estimate_cost
from costdemo.models import UsageNumbers
from costdemo.utils import float_from_env, obj_to_dict

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

logger = logging.getLogger(__name__)


def generate_sample_conversation(num_turns: int) -> list[dict[str, str]]:
    """Generate a long conversation history with realistic messages."""
    messages: list[dict[str, str]] = []
    sample_user_messages = [
        "I'm having trouble with my account. Can you help?",
        "I need to check my invoice from last month.",
        "What's the status of my recent order?",
        "I want to know more about your premium tier.",
        "Can you help me troubleshoot this error?",
        "I'd like to update my billing information.",
        "What are the current shipping times?",
        "I need a refund for my last purchase.",
        "Can you explain your return policy?",
        "I'm interested in subscribing to your service.",
        "Can you confirm my account email address?",
        "I never received my confirmation email after signing up.",
        "Why was I charged twice this billing cycle?",
        "Can I pause my subscription instead of cancelling?",
        "How do I export my account data?",
        "I'd like to upgrade from the standard plan to premium.",
        "Is there a discount for annual billing?",
        "My promo code isn't being accepted at checkout.",
        "Can I add a second user to my account?",
        "I'm moving to a new address — how do I update it?",
    ]
    sample_assistant_messages = [
        "Happy to help. Your account has been active for 18 months on the standard tier. What specific issue can I assist with?",
        "Your July invoice was $89.00 ($49.99 subscription + $39.01 usage), paid on schedule. Want a full breakdown?",
        "Your order is in transit, leaving the fulfillment center July 14th, expected July 18th via ContosoShip (TRK-992188).",
        "Premium is $199/month: priority support, advanced analytics, 1000 API calls/day. Most customers see ROI within a month.",
        "Happy to troubleshoot. What error are you seeing? Common causes: auth failures, rate limiting, misconfigured API keys.",
        "I can update your billing info after identity verification. Go to Account Settings > Billing > Payment Methods.",
        "Standard shipping: 5–7 business days (US), 2–3 days express, 10–14 days international.",
        "We offer 30-day money-back on monthly subscriptions. Let me verify your purchase date and process the refund.",
        "Returns accepted within 30 days in original condition; 7 days for digital products. Free return shipping over $50.",
        "We offer monthly, annual, and multi-year plans with significant savings on longer commitments.",
        "Your account email is alex.parker@example.com. Update it under Account Settings > Profile > Email.",
        "Confirmation emails can take up to 10 minutes or land in spam. I can resend now — just confirm your email address.",
        "I see two $89.00 charges on July 10th — a duplicate billing error. I'll flag it and initiate a refund within 1–2 business days.",
        "Yes, you can pause up to 90 days under Account > Subscription > Pause. Billing resumes automatically when the pause ends.",
        "Export your data from Settings > Privacy > Export Data. You'll receive a download link by email within 24 hours.",
        "Upgrading to Premium takes effect immediately with prorated billing for this cycle. Shall I apply it now?",
        "Annual billing saves 20%: Premium drops from $199 to $159/month, billed as $1,908/year. Want to switch?",
        "Promo codes are case-sensitive and plan-specific. Share the code and I'll check its eligibility against your account.",
        "Business and Premium tiers support multiple users. Invite them under Settings > Team > Invite Member.",
        "Update your address in Account Settings > Profile > Shipping Address. I'll also flag your active shipment for a carrier redirect.",
    ]

    for i in range(num_turns):
        user_msg = sample_user_messages[i % len(sample_user_messages)]
        messages.append({"role": "user", "content": user_msg})
        
        assistant_msg = sample_assistant_messages[i % len(sample_assistant_messages)]
        messages.append({"role": "assistant", "content": assistant_msg})

    return messages


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length using a chars/token heuristic."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    return int(round(len(text) / chars_per_token))


def create_parser() -> argparse.ArgumentParser:
    """Create CLI parser for chat-history bloat simulation."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate one-turn cost impact for an agent request with excessive chat history."
        )
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=20,
        help="Number of conversation turns (user + assistant pairs) in the history.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Input token price in USD per 1M tokens. Falls back to .env when omitted.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Foundry endpoint. Falls back to AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Foundry API key. Falls back to AZURE_OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--api-version",
        default=None,
        help="Azure OpenAI API version. Falls back to AZURE_OPENAI_API_VERSION.",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Model deployment name. Falls back to AZURE_OPENAI_DEPLOYMENT.",
    )
    parser.add_argument(
        "--prompt",
        default="My order is delayed and I was billed twice. Please help me resolve this.",
        help="Current user prompt (latest message in the conversation).",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Output token price in USD per 1M tokens. Falls back to .env when omitted.",
    )
    parser.add_argument(
        "--cached-input-price-per-1m",
        type=float,
        default=None,
        help="Cached input token price in USD per 1M tokens. Falls back to .env when omitted.",
    )
    parser.add_argument(
        "--show-raw-usage",
        action="store_true",
        default=True,
        help="Print the full raw usage payload as JSON.",
    )
    parser.add_argument(
        "--hide-raw-usage",
        action="store_false",
        dest="show_raw_usage",
        help="Hide the raw usage payload JSON section.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    # Show SDK transport calls like: INFO: HTTP Request: POST ... "HTTP/1.1 200 OK"
    logging.getLogger("httpx").setLevel(logging.INFO)


def int_or_default(value: object | None, default: int = 0) -> int:
    """Convert value to int when possible, otherwise return default."""
    if value is None:
        return default
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
            return default
    return default


def get_live_settings(args: argparse.Namespace) -> tuple[str, str, str, str] | None:
    """Resolve live Foundry settings from CLI flags or environment."""
    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = args.api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
    deployment = args.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not endpoint or not api_key or not deployment:
        return None
    return endpoint, api_key, api_version, deployment


def extract_usage(response: object) -> tuple[UsageNumbers, dict[str, object]]:
    """Extract normalized usage counters and raw usage dictionary."""
    usage = getattr(response, "usage", None)
    usage_dict = obj_to_dict(usage)
    return (
        UsageNumbers(
            prompt_tokens=int_or_default(usage_dict.get("prompt_tokens"), 0),
            completion_tokens=int_or_default(usage_dict.get("completion_tokens"), 0),
            total_tokens=int_or_default(usage_dict.get("total_tokens"), 0),
        ),
        usage_dict,
    )


def extract_content(response: object) -> str:
    """Extract first assistant message content from a chat completion response."""
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
        return content
    return ""


def run_live_once(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    messages: list[dict[str, str]],
) -> object:
    """Run one live request with chat history."""
    client = create_client(
        endpoint=endpoint,
        deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=cast(Any, messages),
    )
    return response


def run(args: argparse.Namespace) -> int:
    """Execute one request demo with excessive chat history."""
    if args.history_turns < 0:
        logger.error("--history-turns must be >= 0.")
        return EXIT_ERROR
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

    if input_price_per_1m is None:
        logger.error(
            "Missing input price. Set INPUT_PRICE_PER_1M in .env or pass --input-price-per-1m."
        )
        return EXIT_ERROR
    if input_price_per_1m < 0:
        logger.error("--input-price-per-1m must be >= 0.")
        return EXIT_ERROR

    live_settings = get_live_settings(args)
    if live_settings is None:
        logger.error(
            "Missing live endpoint settings. Provide endpoint, api key, and deployment via CLI flags or .env."
        )
        return EXIT_ERROR

    chat_history = generate_sample_conversation(args.history_turns)
    # Add the current prompt as the final user message
    chat_history.append({"role": "user", "content": args.prompt})

    endpoint, api_key, api_version, deployment = live_settings
    try:
        response = run_live_once(
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment=deployment,
            messages=chat_history,
        )
    except APIError as exc:
        logger.error("Live Foundry request failed: %s", exc)
        return EXIT_FAILURE

    content = extract_content(response)
    usage_numbers, usage_raw = extract_usage(response)
    prompt_tokens_details = obj_to_dict(usage_raw.get("prompt_tokens_details"))
    cached_prompt_tokens = int_or_default(prompt_tokens_details.get("cached_tokens"), 0)

    # Calculate history token contribution
    history_json = json.dumps(chat_history, separators=(",", ":"), sort_keys=True)
    history_tokens_estimated = estimate_token_count(history_json)

    print("\n=== Request Context ===")
    print(f"conversation_turns_in_history: {args.history_turns}")
    print(f"total_messages_in_request: {len(chat_history)}")
    print(f"estimated_history_tokens: {history_tokens_estimated}")

    print("\n=== Full Request Payload ===")
    print(json.dumps({"messages": chat_history}, indent=2))

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
    """CLI entrypoint with robust top-level error handling."""
    load_dotenv()
    args = create_parser().parse_args()
    configure_logging(args.verbose)
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
