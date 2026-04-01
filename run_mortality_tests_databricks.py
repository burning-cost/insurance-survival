"""
Upload insurance-survival to Databricks and run the mortality module tests
with NumPyro/JAX installed via pip.

The mortality tests are separated from the main test runner because:
1. NumPyro + JAX requires ~2GB of additional dependencies
2. MCMC fitting takes several minutes even on small synthetic data
3. Not all Databricks clusters have GPU; CPU is sufficient but slower

Usage:
    python run_mortality_tests_databricks.py
"""

from __future__ import annotations

import os
import sys
import time
import base64
from pathlib import Path


def load_env(path: str) -> None:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


load_env(os.path.expanduser("~/.config/burning-cost/databricks.env"))

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-survival-mortality"
PKG_ROOT = PROJECT_ROOT / "src" / "insurance_survival"


def ensure_dir(remote_path: str) -> None:
    try:
        w.workspace.mkdirs(remote_path)
    except Exception:
        pass


def upload_file(local_path: Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ws_svc.ImportFormat.AUTO,
        overwrite=True,
    )
    print(f"  Uploaded: {remote_path}")


def upload_package_dir(local_dir: Path, remote_dir: str) -> None:
    """Recursively upload a Python package directory."""
    ensure_dir(remote_dir)
    for item in sorted(local_dir.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        remote_item = f"{remote_dir}/{item.name}"
        if item.is_dir() and (item / "__init__.py").exists():
            upload_package_dir(item, remote_item)
        elif item.is_file() and item.suffix == ".py":
            upload_file(item, remote_item)


print("Uploading files to Databricks workspace...")

for subpath in [
    WORKSPACE_PATH,
    f"{WORKSPACE_PATH}/src",
    f"{WORKSPACE_PATH}/src/insurance_survival",
    f"{WORKSPACE_PATH}/tests",
]:
    ensure_dir(subpath)

# Upload full package
upload_package_dir(PKG_ROOT, f"{WORKSPACE_PATH}/src/insurance_survival")

# Upload only mortality tests (faster iteration)
test_dir = PROJECT_ROOT / "tests"
for f in [test_dir / "test_mortality.py"]:
    if f.exists():
        upload_file(f, f"{WORKSPACE_PATH}/tests/{f.name}")

upload_file(PROJECT_ROOT / "pyproject.toml", f"{WORKSPACE_PATH}/pyproject.toml")

print("\nCreating test notebook...")

NOTEBOOK_CONTENT = """# Databricks notebook source
# Install NumPyro + JAX (the mortality[mortality] optional dependencies)
# Plus standard test deps
# MAGIC %pip install \\
# MAGIC     "numpyro>=0.13.0" \\
# MAGIC     "jax>=0.4.25" \\
# MAGIC     "jaxlib>=0.4.25" \\
# MAGIC     "lifelines>=0.29.0" \\
# MAGIC     "polars>=1.0.0" \\
# MAGIC     "scipy>=1.11.0" \\
# MAGIC     "numpy>=1.25" \\
# MAGIC     "pandas>=2.0" \\
# MAGIC     "scikit-learn>=1.3" \\
# MAGIC     "joblib>=1.2" \\
# MAGIC     "matplotlib>=3.7.0" \\
# MAGIC     "pytest>=7.4.0" \\
# MAGIC     "pytest-cov>=4.1.0"

# COMMAND ----------

import sys, os, shutil

# Copy workspace files to /tmp (writable for __pycache__)
for src_dir in ["/Workspace/insurance-survival-mortality/src",
                "/Workspace/insurance-survival-mortality/tests"]:
    dst_dir = src_dir.replace("/Workspace/insurance-survival-mortality", "/tmp/insurance-survival")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

shutil.copy(
    "/Workspace/insurance-survival-mortality/pyproject.toml",
    "/tmp/insurance-survival/pyproject.toml"
)

# COMMAND ----------

import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = "/tmp/insurance-survival/src:/tmp/insurance-survival:/tmp/insurance-survival/tests"
env["PYTHONDONTWRITEBYTECODE"] = "1"
# Suppress JAX verbose logging
env["JAX_LOG_COMPILES"] = "0"
env["TF_CPP_MIN_LOG_LEVEL"] = "3"

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/tmp/insurance-survival/tests/test_mortality.py",
     "-v", "--tb=long", "--no-header",
     "-p", "no:warnings",
     "--import-mode=importlib"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-survival",
    env=env,
    timeout=1800,  # 30 min max for MCMC tests
)

output = result.stdout + "\\nSTDERR:\\n" + result.stderr
if len(output) > 30000:
    output = output[:15000] + "\\n...[middle truncated]...\\n" + output[-15000:]

dbutils.notebook.exit(output)
"""

encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=f"{WORKSPACE_PATH}/run_mortality_tests",
    content=encoded_nb,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print(f"  Uploaded: {WORKSPACE_PATH}/run_mortality_tests")

print("\nSubmitting mortality test job (serverless)...")

result = w.api_client.do("POST", "/api/2.2/jobs/runs/submit", body={
    "run_name": "insurance-survival-mortality-tests",
    "tasks": [{
        "task_key": "run_mortality_tests",
        "notebook_task": {
            "notebook_path": f"{WORKSPACE_PATH}/run_mortality_tests",
        },
    }]
})

run_id = result["run_id"]
host = os.environ["DATABRICKS_HOST"].rstrip("/")
print(f"Job submitted: run_id={run_id}")
print(f"Watch at: {host}#job/runs/{run_id}")

print("\nWaiting for tests (MCMC tests take 10-20 minutes)...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    life_cycle = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    result_state = state.result_state.value if state.result_state else ""
    print(f"  {life_cycle} {result_state}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(60)  # MCMC tests take a while; poll every minute

print("\n" + "=" * 60)

tasks = run_state.tasks or []
task_run_id = None
for t in sorted(tasks, key=lambda x: x.attempt_number or 0):
    task_run_id = t.run_id

if task_run_id is not None:
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output and output.notebook_output.result:
            pytest_output = output.notebook_output.result
            print(pytest_output)
        if output.error:
            print("Notebook error:", output.error)
        if output.error_trace:
            print("Error trace:", output.error_trace[-5000:])
    except Exception as e:
        print(f"Could not retrieve output: {e}")

final_status = result_state or "UNKNOWN"
print(f"\nFinal status: {final_status}")
if final_status != "SUCCESS":
    sys.exit(1)
else:
    print("\nMORTALITY TESTS PASSED")
