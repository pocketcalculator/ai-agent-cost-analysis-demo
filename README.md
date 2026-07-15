---
title: Azure Foundry Token Usage Demo
description: End-to-end tutorial for a multi-script Azure Foundry cost optimization demo repository
author: Microsoft
ms.date: 2026-07-15
ms.topic: tutorial
keywords:
  - azure ai foundry
  - azure openai
  - token usage
  - python
estimated_reading_time: 10
---

## Repository Goal

This repository demonstrates agent-centric cost optimization workflows with reusable Python modules and focused scripts.

The baseline deployment configured in this workspace is:

* Resource group: `rg-foundry-cost-demo`
* Account: `foundrycostdemo42288`
* Project: `cost-agent-demo`
* Deployment name: `cost-helper-chat`
* Model: `gpt-5.4-mini` (`2026-03-17`)
* SKU: `GlobalStandard`

## Directory Structure

```text
ai-agent-cost-analysis-demo/
├─ costdemo/
│  ├─ cli.py
│  ├─ costing.py
│  ├─ foundry_client.py
│  ├─ models.py
│  └─ utils.py
├─ scripts/
│  ├─ agent/
│  │  └─ agent_cost_summary.py
│  ├─ analysis/
│  │  └─ token_cost_breakdown.py
│  ├─ pricing/
│  │  └─ estimate_from_usage_json.py
│  └─ simulation/
│     └─ scenario_cost_simulator.py
├─ main.py
├─ .env.example
├─ pyproject.toml
└─ README.md
```

## Core Command

Primary end-to-end demo:

```bash
~/.local/bin/uv run scripts/agent/agent_cost_summary.py --show-raw-usage
```

Backwards-compatible command:

```bash
~/.local/bin/uv run main.py --show-raw-usage
```

The command prints:

* Model response text
* Prompt tokens
* Completion tokens
* Total tokens
* Prompt cache split (`prompt_tokens_non_cached` and `prompt_tokens_cached`)
* Estimated cost based on env or CLI pricing

## Environment Variables

Copy template and fill values:

```bash
cp .env.example .env
```

Required:

* `AZURE_OPENAI_ENDPOINT`
* `AZURE_OPENAI_API_KEY`
* `AZURE_OPENAI_API_VERSION`
* `AZURE_OPENAI_DEPLOYMENT`

Optional pricing:

* `INPUT_PRICE_PER_1M`
* `OUTPUT_PRICE_PER_1M`
* `CACHED_INPUT_PRICE_PER_1M`

## Additional Demo Scripts

Token totals from saved payload:

```bash
~/.local/bin/uv run scripts/analysis/token_cost_breakdown.py usage.json
```

Cost estimate from saved payload:

```bash
~/.local/bin/uv run scripts/pricing/estimate_from_usage_json.py usage.json --input-price-per-1m 0.75 --output-price-per-1m 4.50
```

Monthly simulation from request assumptions:

```bash
~/.local/bin/uv run scripts/simulation/scenario_cost_simulator.py --requests-per-month 100000 --avg-prompt-tokens 300 --avg-completion-tokens 500 --input-price-per-1m 0.75 --output-price-per-1m 4.50
```

Single-iteration simulation for tool-schema bloat (uses `INPUT_PRICE_PER_1M` from `.env` by default):

```bash
~/.local/bin/uv run scripts/simulation/tool_schema_bloat_simulator.py
```

Realistic mode using your Foundry endpoint (uses `.env` endpoint/key/deployment settings):

```bash
~/.local/bin/uv run scripts/simulation/tool_schema_bloat_simulator.py
```

Optional override if you want to force a price from CLI:

```bash
~/.local/bin/uv run scripts/simulation/tool_schema_bloat_simulator.py --input-price-per-1m 0.75
```

## Naming Conventions

* Keep reusable logic in `costdemo/`.
* Keep one executable concern per file under `scripts/<domain>/`.
* Use descriptive names with `domain_action.py`, for example `token_cost_breakdown.py`.
