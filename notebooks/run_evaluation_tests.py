# Databricks notebook source
# Run tests for insurance_survival.evaluation on Databricks serverless compute.
# This notebook is not part of the demo — it's for CI/test validation.

# COMMAND ----------

# MAGIC %pip install insurance-survival[dev,plot] pytest

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/evaluation/test_censored_eval.py",
     "-v", "--tb=short", "-x"],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
if result.returncode != 0:
    raise RuntimeError(f"Tests failed (exit code {result.returncode})")
