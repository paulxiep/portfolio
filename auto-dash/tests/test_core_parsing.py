"""Tests for plotlint.core.parsing — code and JSON extraction."""

import pytest

from plotlint.core.parsing import parse_code_from_response, parse_json_from_response


class TestParseCode:
    def test_fenced_python_block(self):
        raw = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        assert parse_code_from_response(raw) == "print('hello')"

    def test_fenced_generic_block(self):
        raw = "```\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```"
        assert "import pandas" in parse_code_from_response(raw)

    def test_multiple_blocks_takes_last(self):
        raw = "```python\nfirst\n```\n```python\nsecond\n```"
        assert parse_code_from_response(raw) == "second"

    def test_plain_code_response(self):
        raw = "import matplotlib.pyplot as plt\nplt.plot([1,2,3])"
        result = parse_code_from_response(raw)
        assert "plt.plot" in result

    def test_no_code_raises(self):
        with pytest.raises(ValueError, match="No Python code block"):
            parse_code_from_response("Just some text without any code.")

    def test_strips_whitespace(self):
        raw = "```python\n\n  print('hi')  \n\n```"
        assert parse_code_from_response(raw) == "print('hi')"


class TestParseJson:
    def test_fenced_json_block(self):
        raw = '```json\n{"key": "value"}\n```'
        assert parse_json_from_response(raw) == {"key": "value"}

    def test_plain_json(self):
        raw = '{"a": 1, "b": 2}'
        assert parse_json_from_response(raw) == {"a": 1, "b": 2}

    def test_json_array(self):
        raw = '[1, 2, 3]'
        assert parse_json_from_response(raw) == [1, 2, 3]

    def test_json_in_prose(self):
        raw = 'The result is {"status": "ok"} as expected.'
        result = parse_json_from_response(raw)
        assert result["status"] == "ok"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            parse_json_from_response("No JSON here at all!")

    def test_fenced_generic_block_with_json(self):
        raw = '```\n{"x": 42}\n```'
        assert parse_json_from_response(raw) == {"x": 42}
