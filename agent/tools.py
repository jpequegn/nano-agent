"""Built-in tools for nano-agent: bash, read_file, write_file."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Maximum size (bytes / chars) returned by read_file.
READ_FILE_MAX_CHARS = 8_000

# Timeout in seconds for bash commands.
BASH_TIMEOUT = 10


def _resolve_safe_path(path: str, working_dir: str | None = None) -> Path:
    """Resolve *path* and ensure it stays inside *working_dir*.

    Parameters
    ----------
    path:
        The requested file path (relative or absolute).
    working_dir:
        The directory that acts as the sandbox root.  Defaults to the
        current working directory when *None*.

    Returns
    -------
    Path
        The resolved, validated :class:`~pathlib.Path`.

    Raises
    ------
    ValueError
        If the resolved path would escape *working_dir*.
    """
    root = Path(working_dir or os.getcwd()).resolve()
    target = (root / path).resolve()
    # Path.is_relative_to is available from Python 3.9+
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path '{path}' resolves to '{target}' which is outside the "
            f"working directory '{root}'.  Path traversal is not allowed."
        )
    return target


def bash(command: str, *, working_dir: str | None = None) -> str:
    """Run *command* in a subprocess and return combined stdout + stderr.

    The command is executed with a hard timeout of :data:`BASH_TIMEOUT`
    seconds.  Both stdout and stderr are captured and returned as a single
    string.  If the command times out the output collected so far is
    returned together with a timeout notice.

    Parameters
    ----------
    command:
        Shell command string to execute.
    working_dir:
        Directory in which to run the command.  Defaults to the current
        working directory.

    Returns
    -------
    str
        Combined stdout and stderr output of the command, followed by an
        exit-code line such as ``[exit code: 0]``.
    """
    cwd = working_dir or os.getcwd()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            cwd=cwd,
        )
        output = result.stdout + result.stderr
        return f"{output}[exit code: {result.returncode}]"
    except subprocess.TimeoutExpired as exc:
        collected = ""
        if exc.stdout:
            collected += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
        if exc.stderr:
            collected += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
        return f"{collected}[error: command timed out after {BASH_TIMEOUT}s]"


def read_file(path: str, *, working_dir: str | None = None) -> str:
    """Read the contents of a file, truncating at :data:`READ_FILE_MAX_CHARS` chars.

    Parameters
    ----------
    path:
        Path to the file to read.  Relative paths are resolved against
        *working_dir*.
    working_dir:
        Sandbox root directory.  Path traversal outside this directory is
        rejected.  Defaults to the current working directory.

    Returns
    -------
    str
        File contents (UTF-8, errors replaced), possibly truncated.

    Raises
    ------
    ValueError
        If *path* attempts to escape the working directory.
    FileNotFoundError
        If the resolved path does not exist.
    IsADirectoryError
        If the resolved path is a directory.
    """
    safe_path = _resolve_safe_path(path, working_dir)
    content = safe_path.read_text(encoding="utf-8", errors="replace")
    if len(content) > READ_FILE_MAX_CHARS:
        content = content[:READ_FILE_MAX_CHARS] + f"\n[truncated: file exceeds {READ_FILE_MAX_CHARS} chars]"
    return content


def write_file(path: str, content: str, *, working_dir: str | None = None) -> str:
    """Write *content* to a file, creating parent directories as needed.

    Parameters
    ----------
    path:
        Destination file path.  Relative paths are resolved against
        *working_dir*.
    content:
        Text content to write (UTF-8).
    working_dir:
        Sandbox root directory.  Path traversal outside this directory is
        rejected.  Defaults to the current working directory.

    Returns
    -------
    str
        Confirmation message including the resolved file path and the number
        of bytes written.

    Raises
    ------
    ValueError
        If *path* attempts to escape the working directory.
    """
    safe_path = _resolve_safe_path(path, working_dir)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = safe_path.write_text(content, encoding="utf-8")
    return f"Written {bytes_written} bytes to '{safe_path}'."
