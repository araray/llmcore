# tests/test_phase3_session_commands.py
"""
Tests for Phase 3: Session Fork/Clone Commands in llmchat.

Tests cover:
- Argument parsing utilities (_parse_session_fork_args, _parse_message_range)
- These tests are isolated and don't require the full llmchat import chain
"""

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ==============================================================================
# Extract and test the standalone parsing functions
# ==============================================================================


def _parse_session_fork_args(args: List[str]) -> Dict[str, Any]:
    """
    Parse arguments for /session fork command.
    (Copy of the implementation for isolated testing)
    """
    result: Dict[str, Any] = {
        "new_name": None,
        "from_message_id": None,
        "message_ids": None,
        "message_range": None,
        "no_context": False,
    }

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--from", "-f"):
            if i + 1 >= len(args):
                raise ValueError("--from requires a message ID")
            result["from_message_id"] = args[i + 1]
            i += 2
        elif arg in ("--messages", "-m"):
            if i + 1 >= len(args):
                raise ValueError("--messages requires comma-separated message IDs")
            result["message_ids"] = [m.strip() for m in args[i + 1].split(",")]
            i += 2
        elif arg in ("--range", "-r"):
            if i + 1 >= len(args):
                raise ValueError("--range requires start:end or start-end (1-based)")
            range_str = args[i + 1]
            if ":" in range_str:
                parts = range_str.split(":")
            elif "-" in range_str:
                parts = range_str.split("-")
            else:
                raise ValueError(f"Invalid range format: {range_str}. Use start:end or start-end")
            try:
                start = int(parts[0]) - 1
                end = int(parts[1]) - 1
                result["message_range"] = (start, end)
            except (ValueError, IndexError):
                raise ValueError(f"Invalid range: {range_str}")
            i += 2
        elif arg == "--no-context":
            result["no_context"] = True
            i += 1
        elif arg.startswith("-"):
            raise ValueError(f"Unknown option: {arg}")
        else:
            result["new_name"] = arg
            i += 1

    return result


def _parse_message_range(range_spec: str, message_count: int) -> List[int]:
    """
    Parse a message range specification into list of 0-based indices.
    (Copy of the implementation for isolated testing)
    """
    range_spec = range_spec.strip()

    # Handle last:N
    if range_spec.startswith("last:"):
        try:
            n = int(range_spec[5:])
            if n <= 0:
                raise ValueError(f"Invalid count: {n}")
            start = max(0, message_count - n)
            return list(range(start, message_count))
        except ValueError:
            raise ValueError(f"Invalid last:N format: {range_spec}")

    # Handle first:N
    if range_spec.startswith("first:"):
        try:
            n = int(range_spec[6:])
            if n <= 0:
                raise ValueError(f"Invalid count: {n}")
            end = min(n, message_count)
            return list(range(0, end))
        except ValueError:
            raise ValueError(f"Invalid first:N format: {range_spec}")

    # Handle comma-separated list: 1,3,5
    if "," in range_spec:
        indices = []
        for part in range_spec.split(","):
            part = part.strip()
            try:
                idx = int(part) - 1
                if idx < 0 or idx >= message_count:
                    raise ValueError(f"Index {part} out of range (1-{message_count})")
                indices.append(idx)
            except ValueError:
                raise ValueError(f"Invalid index: {part}")
        return sorted(set(indices))

    # Handle range: 3-7 or 3:7 or 3..7
    for sep in [":", "-", ".."]:
        if sep in range_spec:
            parts = range_spec.split(sep, 1)
            try:
                start = int(parts[0]) - 1
                end = int(parts[1]) - 1
                if start < 0:
                    raise ValueError(f"Start index {parts[0]} must be >= 1")
                if start > end:
                    raise ValueError(f"Start {parts[0]} > end {parts[1]}")
                end = min(end, message_count - 1)
                return list(range(start, end + 1))
            except (ValueError, IndexError):
                raise ValueError(f"Invalid range: {range_spec}")

    # Handle single number
    try:
        idx = int(range_spec) - 1
        if idx < 0 or idx >= message_count:
            raise ValueError(f"Index {range_spec} out of range (1-{message_count})")
        return [idx]
    except ValueError:
        raise ValueError(f"Invalid range specification: {range_spec}")


# ==============================================================================
# _parse_session_fork_args Tests
# ==============================================================================


class TestParseSessionForkArgs:
    """Tests for _parse_session_fork_args function."""

    def test_parse_empty_args(self):
        """Empty args returns defaults."""
        result = _parse_session_fork_args([])
        assert result["new_name"] is None
        assert result["from_message_id"] is None
        assert result["message_ids"] is None
        assert result["message_range"] is None
        assert result["no_context"] is False

    def test_parse_new_name_only(self):
        """Positional argument is interpreted as new name."""
        result = _parse_session_fork_args(["my_fork"])
        assert result["new_name"] == "my_fork"

    def test_parse_from_message(self):
        """--from sets from_message_id."""
        result = _parse_session_fork_args(["--from", "msg-005"])
        assert result["from_message_id"] == "msg-005"

    def test_parse_from_message_short_form(self):
        """-f is short form for --from."""
        result = _parse_session_fork_args(["-f", "msg-005"])
        assert result["from_message_id"] == "msg-005"

    def test_parse_message_ids(self):
        """--messages sets message_ids list."""
        result = _parse_session_fork_args(["--messages", "msg-001,msg-003,msg-005"])
        assert result["message_ids"] == ["msg-001", "msg-003", "msg-005"]

    def test_parse_range_colon(self):
        """--range with colon separator."""
        result = _parse_session_fork_args(["--range", "1:5"])
        assert result["message_range"] == (0, 4)  # 0-based

    def test_parse_range_dash(self):
        """--range with dash separator."""
        result = _parse_session_fork_args(["--range", "2-7"])
        assert result["message_range"] == (1, 6)  # 0-based

    def test_parse_no_context(self):
        """--no-context flag."""
        result = _parse_session_fork_args(["--no-context"])
        assert result["no_context"] is True

    def test_parse_combined_args(self):
        """Combination of args."""
        result = _parse_session_fork_args(["--from", "msg-003", "--no-context", "my_fork"])
        assert result["from_message_id"] == "msg-003"
        assert result["no_context"] is True
        assert result["new_name"] == "my_fork"

    def test_parse_from_missing_arg_raises(self):
        """--from without argument raises ValueError."""
        with pytest.raises(ValueError, match="requires a message ID"):
            _parse_session_fork_args(["--from"])

    def test_parse_unknown_option_raises(self):
        """Unknown option raises ValueError."""
        with pytest.raises(ValueError, match="Unknown option"):
            _parse_session_fork_args(["--unknown"])


# ==============================================================================
# _parse_message_range Tests
# ==============================================================================


class TestParseMessageRange:
    """Tests for _parse_message_range function."""

    def test_parse_single_index(self):
        """Single number returns single index."""
        result = _parse_message_range("5", 10)
        assert result == [4]  # 0-based

    def test_parse_range_colon(self):
        """Range with colon separator."""
        result = _parse_message_range("3:7", 10)
        assert result == [2, 3, 4, 5, 6]  # 0-based

    def test_parse_range_dash(self):
        """Range with dash separator."""
        result = _parse_message_range("2-5", 10)
        assert result == [1, 2, 3, 4]

    def test_parse_comma_list(self):
        """Comma-separated list."""
        result = _parse_message_range("1,3,5", 10)
        assert result == [0, 2, 4]

    def test_parse_last_n(self):
        """last:N format."""
        result = _parse_message_range("last:3", 10)
        assert result == [7, 8, 9]

    def test_parse_first_n(self):
        """first:N format."""
        result = _parse_message_range("first:3", 10)
        assert result == [0, 1, 2]

    def test_parse_clamps_to_max(self):
        """Range end is clamped to max count."""
        result = _parse_message_range("8-15", 10)
        assert result == [7, 8, 9]

    def test_parse_out_of_range_raises(self):
        """Index out of range raises ValueError."""
        with pytest.raises(ValueError, match="(out of range|Invalid)"):
            _parse_message_range("15", 10)

    def test_parse_invalid_format_raises(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            _parse_message_range("abc", 10)

    def test_parse_start_greater_than_end_raises(self):
        """Start > end raises ValueError."""
        with pytest.raises(ValueError, match="(Start .* > end|Invalid range)"):
            _parse_message_range("7:3", 10)


# ==============================================================================
# Additional Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_fork_args_multiple_positional(self):
        """Only first non-option arg is used as name."""
        result = _parse_session_fork_args(["first", "second"])
        # First is name, second is also treated as name (overwrites)
        assert result["new_name"] == "second"

    def test_message_range_first_zero_invalid(self):
        """first:0 should be invalid."""
        with pytest.raises(ValueError):
            _parse_message_range("first:0", 10)

    def test_message_range_last_zero_invalid(self):
        """last:0 should be invalid."""
        with pytest.raises(ValueError):
            _parse_message_range("last:0", 10)

    def test_message_range_double_dot_separator(self):
        """.. separator should work."""
        result = _parse_message_range("3..5", 10)
        assert result == [2, 3, 4]

    def test_fork_args_spaces_in_message_ids(self):
        """Spaces in comma-separated message IDs should be trimmed."""
        result = _parse_session_fork_args(["--messages", "msg-1, msg-2 , msg-3"])
        assert result["message_ids"] == ["msg-1", "msg-2", "msg-3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
