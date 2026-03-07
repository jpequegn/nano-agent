"""Tests for agent.tools: bash, read_file, write_file."""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

from agent.tools import (
    BASH_TIMEOUT,
    READ_FILE_MAX_CHARS,
    bash,
    read_file,
    write_file,
    _resolve_safe_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestResolveSafePath:
    def test_relative_path_within_root(self, tmp_path):
        result = _resolve_safe_path("subdir/file.txt", working_dir=str(tmp_path))
        assert result == tmp_path / "subdir" / "file.txt"

    def test_absolute_path_within_root(self, tmp_path):
        target = tmp_path / "file.txt"
        result = _resolve_safe_path(str(target), working_dir=str(tmp_path))
        assert result == target

    def test_simple_dotdot_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            _resolve_safe_path("../escape.txt", working_dir=str(tmp_path))

    def test_nested_dotdot_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            _resolve_safe_path("subdir/../../escape.txt", working_dir=str(tmp_path))

    def test_absolute_outside_root_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            _resolve_safe_path("/etc/passwd", working_dir=str(tmp_path))

    def test_path_equal_to_root_allowed(self, tmp_path):
        # Resolving to the root itself should be fine.
        result = _resolve_safe_path(".", working_dir=str(tmp_path))
        assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class TestBash:
    def test_echo_stdout(self):
        output = bash("echo hello")
        assert "hello" in output
        assert "[exit code: 0]" in output

    def test_stderr_captured(self):
        output = bash("echo errtext >&2")
        assert "errtext" in output
        assert "[exit code: 0]" in output

    def test_nonzero_exit_code(self):
        output = bash("exit 42", working_dir=os.getcwd())
        assert "[exit code: 42]" in output

    def test_working_dir_respected(self, tmp_path):
        output = bash("pwd", working_dir=str(tmp_path))
        assert str(tmp_path.resolve()) in output

    def test_timeout_returns_error_message(self):
        output = bash(f"sleep {BASH_TIMEOUT + 5}")
        assert "timed out" in output.lower()

    def test_write_and_run_python_file(self, tmp_path):
        """End-to-end: write a Python file, run it, read output via bash."""
        py_file = tmp_path / "hello.py"
        py_file.write_text("print('nano-agent works!')\n", encoding="utf-8")
        output = bash(f"{sys.executable} {py_file}")
        assert "nano-agent works!" in output
        assert "[exit code: 0]" in output


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_simple_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        content = read_file("test.txt", working_dir=str(tmp_path))
        assert content == "hello world"

    def test_read_absolute_path(self, tmp_path):
        f = tmp_path / "abs.txt"
        f.write_text("absolute", encoding="utf-8")
        content = read_file(str(f), working_dir=str(tmp_path))
        assert content == "absolute"

    def test_truncation_at_limit(self, tmp_path):
        big_content = "x" * (READ_FILE_MAX_CHARS + 500)
        f = tmp_path / "big.txt"
        f.write_text(big_content, encoding="utf-8")
        result = read_file("big.txt", working_dir=str(tmp_path))
        assert len(result) > READ_FILE_MAX_CHARS  # includes the truncation notice
        assert "truncated" in result
        assert result[:READ_FILE_MAX_CHARS] == "x" * READ_FILE_MAX_CHARS

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_file("nonexistent.txt", working_dir=str(tmp_path))

    def test_path_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            read_file("../secret.txt", working_dir=str(tmp_path))

    def test_is_a_directory_error(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        with pytest.raises(IsADirectoryError):
            read_file("subdir", working_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_creates_file(self, tmp_path):
        result = write_file("output.txt", "some content", working_dir=str(tmp_path))
        assert (tmp_path / "output.txt").read_text() == "some content"
        assert "Written" in result
        assert "output.txt" in result

    def test_write_reports_byte_count(self, tmp_path):
        content = "hello"
        result = write_file("out.txt", content, working_dir=str(tmp_path))
        expected_bytes = len(content.encode("utf-8"))
        assert str(expected_bytes) in result

    def test_write_creates_parent_dirs(self, tmp_path):
        write_file("a/b/c/deep.txt", "deep", working_dir=str(tmp_path))
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text() == "deep"

    def test_write_overwrites_existing_file(self, tmp_path):
        (tmp_path / "existing.txt").write_text("old")
        write_file("existing.txt", "new content", working_dir=str(tmp_path))
        assert (tmp_path / "existing.txt").read_text() == "new content"

    def test_path_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            write_file("../evil.txt", "bad", working_dir=str(tmp_path))

    def test_write_absolute_outside_root_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            write_file("/tmp/evil_absolute.txt", "bad", working_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Integration: write → run → read
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_write_run_read(self, tmp_path):
        """Write a Python script, execute it, read the output file."""
        script = textwrap.dedent("""\
            result = 6 * 7
            with open("result.txt", "w") as f:
                f.write(f"The answer is {result}\\n")
        """)
        # 1. Write the script
        write_msg = write_file("compute.py", script, working_dir=str(tmp_path))
        assert "Written" in write_msg

        # 2. Run the script with bash
        bash_output = bash(f"{sys.executable} compute.py", working_dir=str(tmp_path))
        assert "[exit code: 0]" in bash_output

        # 3. Read the output file
        result_content = read_file("result.txt", working_dir=str(tmp_path))
        assert "The answer is 42" in result_content
