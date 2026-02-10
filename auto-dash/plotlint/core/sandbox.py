"""Code execution sandbox using subprocess isolation.

Generic subprocess execution used by both plotlint (renderer) and
autodash (explorer). Communication via temp files to avoid stdout
corruption from user code.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from plotlint.core.errors import SandboxError


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    IMPORT_ERROR = "import_error"


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing Python code in a subprocess."""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    return_value: Any = None
    execution_time_ms: int = 0


_WORKER_TEMPLATE = '''
import sys
import pickle
import traceback

# Read the user code
code_path = sys.argv[1]
result_path = sys.argv[2]

with open(code_path, "r", encoding="utf-8") as f:
    code = f.read()

result = {{"status": "success", "error_message": None, "error_type": None, "return_value": None}}

try:
    compiled = compile(code, "<sandbox>", "exec")
    namespace = {{}}
    {inject_block}
    exec(compiled, namespace)
    # If namespace has a __result__ variable, capture it
    if "__result__" in namespace:
        result["return_value"] = namespace["__result__"]
except SyntaxError as e:
    result["status"] = "syntax_error"
    result["error_message"] = str(e)
    result["error_type"] = "SyntaxError"
except ImportError as e:
    result["status"] = "import_error"
    result["error_message"] = str(e)
    result["error_type"] = "ImportError"
except Exception as e:
    result["status"] = "runtime_error"
    result["error_message"] = str(e)
    result["error_type"] = type(e).__name__

with open(result_path, "wb") as f:
    pickle.dump(result, f)
'''


def execute_code(
    code: str,
    timeout_seconds: int = 30,
    allowed_imports: Optional[set[str]] = None,
    inject_globals: Optional[dict[str, Any]] = None,
) -> ExecutionResult:
    """Execute Python code in a subprocess sandbox.

    The subprocess receives code via temp file, executes with timeout,
    returns status and optional return value via temp file.

    Args:
        code: Python source code to execute.
        timeout_seconds: Kill subprocess after this many seconds.
        allowed_imports: Restrict imports (future hardening).
        inject_globals: Variables to inject into execution namespace.
    """
    # Build the inject block for the worker
    inject_block = ""
    inject_file = None
    if inject_globals:
        inject_file = tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False, mode="wb"
        )
        pickle.dump(inject_globals, inject_file)
        inject_file.close()
        inject_block = (
            f'import pickle\n'
            f'    with open(r"{inject_file.name}", "rb") as _inj_f:\n'
            f'        namespace.update(pickle.load(_inj_f))'
        )

    worker_code = _WORKER_TEMPLATE.format(inject_block=inject_block)

    # Write user code and worker to temp files
    code_file = tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    )
    code_file.write(code)
    code_file.close()

    worker_file = tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    )
    worker_file.write(worker_code)
    worker_file.close()

    result_file = tempfile.NamedTemporaryFile(
        suffix=".pkl", delete=False, mode="wb"
    )
    result_file.close()

    start_time = time.monotonic()

    try:
        proc = subprocess.run(
            [sys.executable, worker_file.name, code_file.name, result_file.name],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Read the result from the temp file
        try:
            with open(result_file.name, "rb") as f:
                result_data = pickle.load(f)
        except (EOFError, FileNotFoundError, pickle.UnpicklingError):
            # Worker crashed before writing result
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                stdout=proc.stdout,
                stderr=proc.stderr,
                error_message="Worker process crashed before writing result",
                execution_time_ms=elapsed_ms,
            )

        status = ExecutionStatus(result_data["status"])
        return ExecutionResult(
            status=status,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error_message=result_data.get("error_message"),
            error_type=result_data.get("error_type"),
            return_value=result_data.get("return_value"),
            execution_time_ms=elapsed_ms,
        )

    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            error_message=f"Execution timed out after {timeout_seconds}s",
            execution_time_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        raise SandboxError(f"Sandbox execution failed: {e}") from e

    finally:
        # Clean up temp files
        for path in [code_file.name, worker_file.name, result_file.name]:
            try:
                os.unlink(path)
            except OSError:
                pass
        if inject_file:
            try:
                os.unlink(inject_file.name)
            except OSError:
                pass
