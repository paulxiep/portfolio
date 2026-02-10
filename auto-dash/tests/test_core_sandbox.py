"""Tests for plotlint.core.sandbox — code execution."""

from plotlint.core.sandbox import ExecutionResult, ExecutionStatus, execute_code


def test_success():
    result = execute_code("x = 1 + 1")
    assert result.status == ExecutionStatus.SUCCESS
    assert result.error_message is None


def test_stdout_captured():
    result = execute_code("print('hello world')")
    assert result.status == ExecutionStatus.SUCCESS
    assert "hello world" in result.stdout


def test_syntax_error():
    result = execute_code("def bad(")
    assert result.status == ExecutionStatus.SYNTAX_ERROR
    assert result.error_message is not None


def test_runtime_error():
    result = execute_code("x = 1 / 0")
    assert result.status == ExecutionStatus.RUNTIME_ERROR
    assert "ZeroDivisionError" in (result.error_type or "")


def test_import_error():
    result = execute_code("import nonexistent_module_xyz")
    assert result.status == ExecutionStatus.IMPORT_ERROR


def test_timeout():
    result = execute_code("import time; time.sleep(10)", timeout_seconds=1)
    assert result.status == ExecutionStatus.TIMEOUT


def test_return_value():
    result = execute_code("__result__ = 42")
    assert result.status == ExecutionStatus.SUCCESS
    assert result.return_value == 42


def test_execution_time_tracked():
    result = execute_code("x = 1")
    assert result.execution_time_ms >= 0
