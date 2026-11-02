"""Archived executable records are data; maintained tests live in tests/nnao."""
from pathlib import Path


collect_ignore=[p.name for p in Path(__file__).parent.iterdir() if p.is_dir()]
