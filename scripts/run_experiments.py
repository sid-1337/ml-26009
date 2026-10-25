#!/usr/bin/env python3
"""Reproduce every assignment experiment; see README for seed and provenance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task99.experiments import main

if __name__ == '__main__':
    main()
