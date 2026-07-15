# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from costdemo.models import UsageNumbers


def estimate_cost(
    usage: UsageNumbers,
    cached_prompt_tokens: int,
    input_price_per_1m: float | None,
    cached_input_price_per_1m: float | None,
    output_price_per_1m: float | None,
) -> dict[str, float]:
    """Estimate usage cost with optional cached-input pricing support."""
    if input_price_per_1m is None or output_price_per_1m is None:
        return {}

    prompt_tokens = usage.prompt_tokens or 0
    non_cached_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)
    completion_tokens = usage.completion_tokens or 0

    effective_cached_input_price = (
        cached_input_price_per_1m
        if cached_input_price_per_1m is not None
        else input_price_per_1m
    )

    input_cost_non_cached = (non_cached_prompt_tokens / 1_000_000) * input_price_per_1m
    input_cost_cached = (cached_prompt_tokens / 1_000_000) * effective_cached_input_price
    input_cost = input_cost_non_cached + input_cost_cached
    output_cost = (completion_tokens / 1_000_000) * output_price_per_1m

    return {
        "prompt_tokens_non_cached": float(non_cached_prompt_tokens),
        "prompt_tokens_cached": float(cached_prompt_tokens),
        "input_cost_non_cached_usd": input_cost_non_cached,
        "input_cost_cached_usd": input_cost_cached,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }
