#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the repository root is importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from costdemo.cli import main


if __name__ == "__main__":
    sys.exit(main())
