#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, cast

from dotenv import load_dotenv
from openai import APIError

# Ensure the repository root is importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from costdemo.foundry_client import create_client
from scripts.simulation.tool_schema_bloat_simulator import (
    create_tool_schema,
    execute_mock_tool,
    extract_usage,
    get_live_settings,
    parse_tool_calls,
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI parser for A/B tool schema measurement."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare prompt token usage for a full tool schema against a fixed smaller tool schema."
        )
    )
    parser.add_argument(
        "--tool-count",
        type=int,
        default=10,
        help="Number of tools to load into the baseline request schema.",
    )
    parser.add_argument(
        "--comparison-tool-count",
        type=int,
        default=3,
        help="Number of tools to include in the deterministic comparison schema.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "My order is delayed and I was billed twice. "
            "Please check my customer profile, shipping status, and latest invoice, "
            "then explain next steps and open a support ticket if needed."
        ),
        help="Prompt used for both A and B runs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to a .env file. Defaults to the repository root .env file.",
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
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between sequential API calls. Useful for high tool counts.",
    )
    return parser


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.INFO)


def load_environment(env_file: Path | None) -> None:
    """Load environment variables from the provided or default .env file."""
    repo_root = Path(__file__).resolve().parents[2]
    dotenv_path = env_file or repo_root / ".env"
    load_dotenv(dotenv_path)


def prompt_tokens(response: object) -> int:
    """Return prompt token usage from a chat completion response."""
    usage, _ = extract_usage(response)
    return int(usage.prompt_tokens or 0)


def log_prompt_tokens(label: str, response: object) -> int:
    """Log and return prompt token usage for a response."""
    tokens = prompt_tokens(response)
    logger.info("%s prompt_tokens=%s", label, tokens)
    return tokens


def maybe_pause(seconds: float, *, after_label: str) -> None:
    """Pause between API calls when requested."""
    if seconds <= 0:
        return
    logger.info("Pausing %.1f seconds after %s", seconds, after_label)
    time.sleep(seconds)


def first_call(*, client: Any, deployment: str, prompt: str, tools: list[dict[str, Any]]) -> object:
    """Run the first call that lets the model choose tool calls."""
    return client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise cloud cost analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        tools=cast(Any, tools),
        tool_choice="auto",
    )


def build_second_call_messages(
    *, prompt: str, first_response: object
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Build second-turn messages and capture which tools were used.

    Args:
        prompt: User prompt used in the first call.
        first_response: Response object from the first model call.

    Returns:
        Tuple of parsed tool calls, second-call messages, and used tool names.
    """
    tool_calls = parse_tool_calls(first_response)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a concise cloud cost analysis assistant."},
        {"role": "user", "content": prompt},
    ]
    if not tool_calls:
        return [], messages, []

    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        }
    )

    used_names: list[str] = []
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
        result = execute_mock_tool(function_name, parsed_args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(call.get("id") or ""),
                "name": function_name,
                "content": json.dumps(result),
            }
        )
        used_names.append(function_name)

    return tool_calls, messages, used_names


def second_call(*, client: Any, deployment: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> object:
    """Run the second call that synthesizes the final response."""
    return client.chat.completions.create(
        model=deployment,
        messages=cast(Any, messages),
        tools=cast(Any, tools),
        tool_choice="none",
    )


def run(args: argparse.Namespace) -> int:
    """Execute A/B measurement for unused tool schema overhead."""
    if args.tool_count <= 0:
        logger.error("--tool-count must be > 0.")
        return EXIT_ERROR
    if args.comparison_tool_count <= 0:
        logger.error("--comparison-tool-count must be > 0.")
        return EXIT_ERROR
    if args.comparison_tool_count > args.tool_count:
        logger.error("--comparison-tool-count must be <= --tool-count.")
        return EXIT_ERROR
    if args.tool_count > 128:
        logger.error("--tool-count must be <= 128 for this API.")
        return EXIT_ERROR

    load_environment(args.env_file)
    live_settings = get_live_settings(args)
    if live_settings is None:
        logger.error(
            "Missing live endpoint settings. Provide endpoint, api key, and deployment via CLI flags or .env."
        )
        return EXIT_ERROR

    endpoint, api_key, api_version, deployment = live_settings
    client = create_client(
        endpoint=endpoint,
        deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )

    full_tools = create_tool_schema(args.tool_count)
    comparison_tools = full_tools[: args.comparison_tool_count]
    comparison_tool_names = [
        str(tool.get("function", {}).get("name", "")) for tool in comparison_tools
    ]

    try:
        first_response_full = first_call(
            client=client,
            deployment=deployment,
            prompt=args.prompt,
            tools=full_tools,
        )
        first_prompt_tokens_full = log_prompt_tokens("first_call_full", first_response_full)

        tool_calls, second_messages, used_names = build_second_call_messages(
            prompt=args.prompt,
            first_response=first_response_full,
        )
        if not tool_calls:
            print(
                json.dumps(
                    {
                        "used_tool_names": [],
                        "counts": {
                            "full_tools": len(full_tools),
                            "used_only_tools": 0,
                        },
                        "prompt_tokens": {
                            "first_call_full": first_prompt_tokens_full,
                        },
                        "note": "The model did not invoke any tools in the baseline run.",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return EXIT_SUCCESS

        maybe_pause(args.pause_seconds, after_label="first_call_full")

        second_response_full = second_call(
            client=client,
            deployment=deployment,
            messages=second_messages,
            tools=full_tools,
        )
        second_prompt_tokens_full = log_prompt_tokens("second_call_full", second_response_full)

        maybe_pause(args.pause_seconds, after_label="second_call_full")

        first_response_comparison = first_call(
            client=client,
            deployment=deployment,
            prompt=args.prompt,
            tools=comparison_tools,
        )
        first_prompt_tokens_comparison = log_prompt_tokens(
            "first_call_comparison", first_response_comparison
        )

        maybe_pause(args.pause_seconds, after_label="first_call_comparison")

        second_response_comparison = second_call(
            client=client,
            deployment=deployment,
            messages=second_messages,
            tools=comparison_tools,
        )
        second_prompt_tokens_comparison = log_prompt_tokens(
            "second_call_comparison", second_response_comparison
        )
    except APIError as exc:
        logger.error("Live Foundry request failed: %s", exc)
        return EXIT_FAILURE

    result = {
        "used_tool_names": used_names,
        "counts": {
            "full_tools": len(full_tools),
            "comparison_tools": len(comparison_tools),
        },
        "comparison_tool_names": comparison_tool_names,
        "prompt_tokens": {
            "first_call_full": first_prompt_tokens_full,
            "first_call_comparison": first_prompt_tokens_comparison,
            "first_call_tool_overhead_delta": (
                first_prompt_tokens_full - first_prompt_tokens_comparison
            ),
            "second_call_full": second_prompt_tokens_full,
            "second_call_comparison": second_prompt_tokens_comparison,
            "second_call_tool_overhead_delta": (
                second_prompt_tokens_full - second_prompt_tokens_comparison
            ),
            "total_tool_overhead_delta": (
                (first_prompt_tokens_full - first_prompt_tokens_comparison)
                + (second_prompt_tokens_full - second_prompt_tokens_comparison)
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_SUCCESS


def main() -> int:
    """CLI entrypoint with robust top-level error handling."""
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
