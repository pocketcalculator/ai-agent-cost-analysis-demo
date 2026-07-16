---
title: Token Consumption Management Simulators
description: Focused guide for running simulator scripts that explain and control token usage
author: Paul Sczurek
ms.date: 2026-07-16
ms.topic: tutorial
keywords:
  - token consumption
  - simulation
  - azure ai foundry
  - gemini
  - python
estimated_reading_time: 8
---

## Goal

This repository contains practical simulations that help you understand and control token usage patterns in AI-agent workflows.

This README focuses on four scripts:

* [scripts/simulation/tool_schema_ab_measurement.py](scripts/simulation/tool_schema_ab_measurement.py)
* [scripts/simulation/chat_history_bloat_simulator.py](scripts/simulation/chat_history_bloat_simulator.py)
* [scripts/simulation/circuit_breaker_simulator.py](scripts/simulation/circuit_breaker_simulator.py)
* [scripts/simulation/manager_worker_routing_simulator.py](scripts/simulation/manager_worker_routing_simulator.py)

## Script Summaries

### Run tool_schema_ab_measurement.py

Compares prompt-token overhead between a larger tool schema and a smaller comparison schema using live Azure Foundry calls.

What it demonstrates:

* How unused tools inflate prompt tokens
* First-call and second-call token deltas
* Total overhead from schema size alone

### Run chat_history_bloat_simulator.py

Simulates one live request with long multi-turn conversation history to show how stale context increases prompt tokens and cost.

What it demonstrates:

* Prompt growth from accumulated chat turns
* One-turn cost impact from historical context bloat
* Token and cost reporting using live response usage

### Run circuit_breaker_simulator.py

Implements a pre-flight token guard that counts tokens before generation and blocks overly large requests.

What it demonstrates:

* Prompt token counting before generation
* Budget threshold enforcement (circuit breaker)
* Actual usage reporting after successful calls
* Fallback model behavior for common API availability or quota conditions

### Run manager_worker_routing_simulator.py

Simulates manager-worker routing where a cheaper model triages request complexity and escalates complex tasks to a stronger model.

What it demonstrates:

* Tiered model orchestration for cost control
* Retry behavior under throttling and temporary service issues
* Fast-fail behavior with bounded timeout and retry attempts
* Graceful handling when model responses are unavailable

## Environment Setup

### Option 1: uv (recommended)

From the repository root:

```bash
cp .env.example .env
~/.local/bin/uv sync
```

### Option 2: pip and venv

From the repository root:

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Environment Variables

Set values in the .env file before running scripts.

### For tool_schema_ab_measurement.py and chat_history_bloat_simulator.py

Required:

* AZURE_OPENAI_ENDPOINT
* AZURE_OPENAI_API_KEY
* AZURE_OPENAI_API_VERSION
* AZURE_OPENAI_DEPLOYMENT

For chat_history_bloat_simulator.py, also provide one of these:

* INPUT_PRICE_PER_1M
* AZURE_OPENAI_INPUT_PRICE_PER_1M

### For circuit_breaker_simulator.py and manager_worker_routing_simulator.py

Required:

* GEMINI_API_KEY

Optional:

* GEMINI_MODEL

## Run Commands

### tool_schema_ab_measurement.py

With uv:

```bash
~/.local/bin/uv run scripts/simulation/tool_schema_ab_measurement.py
```

With pip and activated venv:

```bash
python scripts/simulation/tool_schema_ab_measurement.py
```

Example with explicit schema sizes:

```bash
~/.local/bin/uv run scripts/simulation/tool_schema_ab_measurement.py --tool-count 10 --comparison-tool-count 3
```

### chat_history_bloat_simulator.py

With uv:

```bash
~/.local/bin/uv run scripts/simulation/chat_history_bloat_simulator.py
```

With pip and activated venv:

```bash
python scripts/simulation/chat_history_bloat_simulator.py
```

Example with an explicit history length:

```bash
~/.local/bin/uv run scripts/simulation/chat_history_bloat_simulator.py --history-turns 40
```

### circuit_breaker_simulator.py

With uv:

```bash
~/.local/bin/uv run scripts/simulation/circuit_breaker_simulator.py
```

With pip and activated venv:

```bash
python scripts/simulation/circuit_breaker_simulator.py
```

### manager_worker_routing_simulator.py

With uv:

```bash
~/.local/bin/uv run scripts/simulation/manager_worker_routing_simulator.py
```

With pip and activated venv:

```bash
python scripts/simulation/manager_worker_routing_simulator.py
```

## Notes

* If you hit 429 quota errors, the Gemini-based simulators may return no model output while still completing the script flow.
* manager_worker_routing_simulator.py uses bounded retries and a request timeout to avoid long apparent hangs.
