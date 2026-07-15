# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from openai import OpenAI


def create_client(endpoint: str, deployment: str, api_key: str, api_version: str) -> OpenAI:
    """Create an Azure OpenAI client for a specific Foundry deployment."""
    base_url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_query={"api-version": api_version},
    )


def run_chat_completion(client: OpenAI, deployment: str, prompt: str) -> object:
    """Run a single chat completion request using the configured deployment."""
    return client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "You are a concise cloud cost analysis assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )
