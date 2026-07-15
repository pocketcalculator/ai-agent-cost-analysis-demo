# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def obj_to_dict(value: object | None) -> dict[str, object]:
    """Convert SDK model objects or dict-like values to plain dicts."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return {}


def float_from_env(*names: str) -> float | None:
    """Return the first valid float from the given environment variable names."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return float(raw)
        except ValueError:
            logger.warning("Ignoring invalid float in %s: %s", name, raw)
    return None
