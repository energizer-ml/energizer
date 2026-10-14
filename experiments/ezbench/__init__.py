"""Measurement harness for the Energizer paper.

Every number reported in the paper must be regenerable with the scripts in
``experiments/``. Raw measurements are written as JSON Lines under ``results/``;
tables and figures are derived from those files only (``make_tables.py``).
"""

__all__ = ["env", "models", "coreml_utils", "power", "timing", "results"]
