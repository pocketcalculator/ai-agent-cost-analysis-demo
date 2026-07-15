# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsageNumbers:
    """Normalized usage counters from a chat completion response."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
