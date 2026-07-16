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

MOCK_CUSTOMER_ID = "cust_10027"

DEFAULT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "crm_lookup_customer",
        "description": "Lookup customer profile and account status.",
    },
    {
        "name": "billing_get_invoice",
        "description": "Fetch invoice details for a customer account.",
    },
    {
        "name": "billing_submit_adjustment",
        "description": "Submit a billing adjustment request.",
    },
    {
        "name": "inventory_get_stock",
        "description": "Return current stock for a product SKU.",
    },
    {
        "name": "order_create_return",
        "description": "Create a return authorization for an order.",
    },
    {
        "name": "order_get_shipping_status",
        "description": "Get the current shipping status and ETA.",
    },
    {
        "name": "hr_fetch_policy",
        "description": "Retrieve policy text from the HR policy catalog.",
    },
    {
        "name": "it_open_service_ticket",
        "description": "Open an internal IT service desk ticket.",
    },
    {
        "name": "compliance_get_control",
        "description": "Retrieve control requirements for compliance checks.",
    },
    {
        "name": "finance_forecast_expense",
        "description": "Estimate upcoming monthly spend for a cost center.",
    },
]


def create_tool_schema(tool_count: int) -> list[dict[str, Any]]:
    """Build function tool list sized by requested tool count.

    Args:
        tool_count: Number of tools to include in the request payload.

    Returns:
        List of function tool descriptors.
    """
    tools: list[dict[str, Any]] = []
    for i in range(tool_count):
        if i < len(DEFAULT_TOOL_DEFINITIONS):
            tool_name = str(DEFAULT_TOOL_DEFINITIONS[i]["name"])
            tool_description = str(DEFAULT_TOOL_DEFINITIONS[i]["description"])
        else:
            tool_name = f"legacy_system_utility_{i}"
            tool_description = "A simulated legacy tool for general system management."
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "region": {"type": "string"},
                            "dry_run": {"type": "boolean"},
                        },
                        "required": ["id"],
                    },
                },
            }
        )
    return tools


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length using a chars/token heuristic."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    return int(round(len(text) / chars_per_token))


def create_parser() -> argparse.ArgumentParser:
    """Create CLI parser for tool-schema bloat simulation."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate one-turn cost impact for an agent request with a configurable number of tools."
        )
    )
    parser.add_argument(
        "--tool-count",
        type=int,
        default=10,
        help="Number of tools loaded into the request schema.",
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
        default=(
            "My order is delayed and I was billed twice. "
            "Please check my customer profile, shipping status, and latest invoice, "
            "then explain next steps and open a support ticket if needed."
        ),
        help="Prompt used for live mode request.",
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
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print loaded tool names for the request.",
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


def parse_tool_calls(response: object) -> list[dict[str, Any]]:
    """Extract tool calls from the first assistant message in a response."""
    response_obj = response if isinstance(response, dict) else getattr(response, "__dict__", {})
    choices: Any = getattr(response, "choices", None)
    if choices is None and isinstance(response_obj, dict):
        choices = response_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return []

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")

    tool_calls: Any = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")

    parsed: list[dict[str, Any]] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            call_dict = obj_to_dict(call)
            if not call_dict and isinstance(call, dict):
                call_dict = call
            if call_dict:
                parsed.append(call_dict)
    return parsed


def execute_mock_tool(function_name: str, function_args: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic mock data for known tools.

    This keeps the demo self-contained while still showing realistic tool-usage flow.
    """
    customer_id = str(function_args.get("id") or MOCK_CUSTOMER_ID)

    if function_name == "crm_lookup_customer":
        return {
            "customer_id": customer_id,
            "name": "Alex Parker",
            "tier": "gold",
            "status": "active",
            "risk_flags": [],
        }
    if function_name == "order_get_shipping_status":
        return {
            "order_id": "ord_778921",
            "shipping_status": "delayed",
            "eta": "2026-07-18",
            "carrier": "ContosoShip",
            "tracking_id": "TRK-992188",
        }
    if function_name == "billing_get_invoice":
        return {
            "customer_id": customer_id,
            "invoice_id": "inv_2026_07_10027",
            "double_charge_detected": True,
            "amount_usd": 89.0,
            "currency": "USD",
        }
    if function_name == "billing_submit_adjustment":
        return {
            "customer_id": customer_id,
            "adjustment_id": "adj_90172",
            "status": "submitted",
            "reason": "duplicate_charge",
            "expected_refund_usd": 89.0,
            "sla_hours": 24,
        }
    if function_name == "inventory_get_stock":
        return {
            "sku": "sku_wireless_headset_01",
            "region": str(function_args.get("region") or "us-east"),
            "in_stock": True,
            "available_quantity": 42,
            "restock_eta_days": 0,
        }
    if function_name == "order_create_return":
        return {
            "return_id": "ret_33019",
            "order_id": "ord_778921",
            "status": "approved",
            "dropoff_required": True,
            "label_url": "https://contoso.example/returns/ret_33019/label.pdf",
        }
    if function_name == "it_open_service_ticket":
        return {
            "ticket_id": "it_54021",
            "status": "opened",
            "queue": "customer-billing",
        }
    if function_name == "hr_fetch_policy":
        return {
            "policy_id": "hr_escalation_001",
            "title": "Customer Escalation Policy",
            "summary": "Escalate unresolved billing disputes after one business day.",
            "version": "2026.2",
        }
    if function_name == "compliance_get_control":
        return {
            "control_id": "cmp_billing_audit_04",
            "framework": "SOC2",
            "requirement": "Track adjustments and preserve audit trail for 12 months.",
            "status": "compliant",
        }
    if function_name == "finance_forecast_expense":
        return {
            "cost_center": "support-operations",
            "forecast_period": "next_30_days",
            "forecast_usd": 12450.0,
            "confidence": 0.86,
            "trend": "stable",
        }
    return {
        "tool": function_name,
        "status": "executed",
        "note": "Mock result generated for demo.",
    }


def run_live_once(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    prompt: str,
    tools: list[dict[str, Any]],
) -> tuple[object, list[str], list[dict[str, Any]]]:
    """Run one live request, execute mock tool calls, and return final response, tool names, and results."""
    client = create_client(
        endpoint=endpoint,
        deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )
    first_response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a retail customer support operator."},
            {"role": "user", "content": prompt},
        ],
        tools=cast(Any, tools),
        tool_choice="auto",
    )

    tool_calls = parse_tool_calls(first_response)
    if not tool_calls:
        return first_response, [], []

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a retail customer support operator."},
        {"role": "user", "content": prompt},
    ]

    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    messages.append(assistant_message)

    executed_tool_names: list[str] = []
    tool_results: list[dict[str, Any]] = []
    for call in tool_calls:
        function_obj = call.get("function") or {}
        function_name = str(function_obj.get("name") or "unknown_tool")
        raw_args = function_obj.get("arguments")
        parsed_args: dict[str, Any] = {}
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                maybe_args = json.loads(raw_args)
                if isinstance(maybe_args, dict):
                    parsed_args = maybe_args
            except json.JSONDecodeError:
                parsed_args = {}

        tool_result = execute_mock_tool(function_name, parsed_args)
        tool_results.append({
            "function_name": function_name,
            "arguments": parsed_args,
            "result": tool_result,
        })
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(call.get("id") or ""),
                "name": function_name,
                "content": json.dumps(tool_result),
            }
        )
        executed_tool_names.append(function_name)

    final_response = client.chat.completions.create(
        model=deployment,
        messages=cast(Any, messages),
        tools=cast(Any, tools),
        tool_choice="none",
    )

    return final_response, executed_tool_names, tool_results


def run(args: argparse.Namespace) -> int:
    """Execute one request demo with a configurable number of tools."""
    if args.tool_count <= 0:
        logger.error("--tool-count must be > 0.")
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

    tools_schema = create_tool_schema(args.tool_count)
    endpoint, api_key, api_version, deployment = live_settings
    try:
        response, executed_tools, tool_results = run_live_once(
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment=deployment,
            prompt=args.prompt,
            tools=tools_schema,
        )
    except APIError as exc:
        logger.error("Live Foundry request failed: %s", exc)
        return EXIT_FAILURE

    content = extract_content(response)
    usage_numbers, usage_raw = extract_usage(response)
    prompt_tokens_details = obj_to_dict(usage_raw.get("prompt_tokens_details"))
    cached_prompt_tokens = int_or_default(prompt_tokens_details.get("cached_tokens"), 0)

    print("\n=== Request Context ===")
    print(f"tools_loaded_in_request: {len(tools_schema)}")
    print(f"tools_invoked: {len(executed_tools)}")
    if executed_tools:
        print(f"invoked_tool_names: {', '.join(executed_tools)}")

    print("\n=== Raw Prompt ===")
    print(args.prompt)

    print("\n=== All Tool Definitions Loaded in Schema ===")
    print(json.dumps(tools_schema, indent=2))

    if tool_results:
        print("\n=== Tool Invocations and Results ===")
        for i, tool_data in enumerate(tool_results, 1):
            print(f"\nTool {i}: {tool_data['function_name']}")
            if tool_data['arguments']:
                print(f"Arguments: {json.dumps(tool_data['arguments'], indent=2)}")
            print(f"Result: {json.dumps(tool_data['result'], indent=2)}")

    invoked_tool_name_set = set(executed_tools)
    unused_tools = [
        tool for tool in tools_schema if str(tool.get("function", {}).get("name", "")) not in invoked_tool_name_set
    ]
    unused_schema_json = json.dumps(unused_tools, separators=(",", ":"), sort_keys=True)
    unused_schema_tokens_estimated = estimate_token_count(unused_schema_json)
    print(f"unused_tools_loaded: {len(unused_tools)}")
    print(f"unused_input_tokens_per_turn_estimated: {unused_schema_tokens_estimated}")

    if args.show_tools:
        tool_names = ", ".join(t["function"]["name"] for t in tools_schema)
        print(f"tools: {tool_names}")

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