from __future__ import annotations

import asyncio
import dataclasses
import shlex
import subprocess
import sys
from logging import getLogger as get_logger
from subprocess import CompletedProcess

from milatools.cli import console
from milatools.utils.remote_v1 import Hide
from milatools.utils.runner import Runner

logger = get_logger(__name__)


@dataclasses.dataclass(init=False, frozen=True)
class LocalV2(Runner):
    """A runner that runs commands in subprocesses on the local machine."""

    hostname = "localhost"

    @staticmethod
    def run(
        command: str | tuple[str, ...],
        input: str | None = None,
        display: bool = True,
        warn: bool = False,
        hide: Hide = False,
        _stack_level: int = 2,
    ) -> CompletedProcess[str]:
        program_and_args = _display_command(
            command, input=input, display=display, _stack_level=_stack_level + 1
        )
        # Using a stacklevel of 3 so that clicking on the source of the logs takes us to the
        # call to `run` or `run_async` with the command, which is much more informative than
        # if it were to lead us here.
        return run(
            program_and_args=program_and_args,
            input=input,
            warn=warn,
            hide=hide,
            _stack_level=_stack_level + 1,
        )

    @staticmethod
    def get_output(
        command: str | tuple[str, ...],
        *,
        display: bool = False,
        warn: bool = False,
        hide: Hide = True,
    ) -> str:
        return LocalV2.run(
            command, display=display, warn=warn, hide=hide
        ).stdout.strip()

    @staticmethod
    async def run_async(
        command: str | tuple[str, ...],
        input: str | None = None,
        display: bool = True,
        warn: bool = False,
        hide: Hide = False,
        _stack_level: int = 2,
    ) -> CompletedProcess[str]:
        program_and_args = _display_command(
            command, input=input, display=display, _stack_level=_stack_level + 1
        )
        return await run_async(
            program_and_args,
            input=input,
            warn=warn,
            hide=hide,
            _stack_level=_stack_level + 1,
        )

    @staticmethod
    async def get_output_async(
        command: str | tuple[str, ...],
        *,
        display: bool = False,
        warn: bool = False,
        hide: Hide = True,
    ) -> str:
        return (
            await LocalV2.run_async(command, display=display, warn=warn, hide=hide)
        ).stdout.strip()


def _display_command(
    command: str | tuple[str, ...],
    input: str | None,
    display: bool,
    _stack_level: int = 2,
) -> tuple[str, ...]:
    """Converts the command to a tuple of strings if needed with `shlex.split` and
    optionally logs it to the console.

    Also shows the input that would be passed to the command, if any.
    """
    if isinstance(command, str):
        program_and_args = tuple(shlex.split(command))
        displayed_command = command
    else:
        program_and_args = command
        displayed_command = shlex.join(command)
    if display:
        if not input:
            console.log(
                f"(localhost) $ {displayed_command}",
                style="green",
                _stack_offset=_stack_level,
            )
        else:
            console.log(
                f"(localhost) $ {displayed_command}\n{input}",
                style="green",
                _stack_offset=_stack_level,
            )
    return program_and_args


def run(
    program_and_args: tuple[str, ...],
    input: str | None = None,
    warn: bool = False,
    hide: Hide = False,
    _stack_level: int = 2,
) -> subprocess.CompletedProcess[str]:
    """Runs the command *synchronously* in a subprocess and returns the result.

    Parameters
    ----------
    program_and_args: The program and arguments to pass to it. This is a tuple of \
        strings, same as in `subprocess.Popen`.
    input: The optional 'input' argument to `subprocess.Popen.communicate()`.
    warn: When `True` and an exception occurs, warn instead of raising the exception.
    hide: Controls the printing of the subprocess' stdout and stderr.

    Returns
    -------
    The `subprocess.CompletedProcess` object with the result of the subprocess.

    Raises
    ------
    subprocess.CalledProcessError
        If an error occurs when running the command and `warn` is `False`.
    """
    if not input:
        logger.debug(f"Calling `subprocess.run` with {program_and_args=}")
    else:
        logger.debug(f"Calling `subprocess.run` with {program_and_args=} and {input=}")
    result = subprocess.run(
        program_and_args,
        shell=False,
        capture_output=True,
        text=True,
        check=not warn,
        input=input,
    )
    assert result.returncode is not None
    # Note: in the case of `check=True` (warn=False), the exception will have already
    # been raised by `subprocess.run` here.
    _warn_or_raise_on_error(
        result=result,
        input=input,
        warn=warn,
        hide=hide,
        _stack_level=_stack_level + 1,
    )
    _print_stdout_stderr(result, hide=hide, _stack_level=_stack_level + 1)
    return result


async def run_async(
    program_and_args: tuple[str, ...],
    input: str | None = None,
    warn: bool = False,
    hide: Hide = False,
    _stack_level: int = 2,
) -> subprocess.CompletedProcess[str]:
    """Runs the command *asynchronously* in a subprocess and returns the result.

    Parameters
    ----------
    program_and_args: The program and arguments to pass to it. This is a tuple of \
        strings, same as in `subprocess.Popen`.
    input: The optional 'input' argument to `subprocess.Popen.communicate()`.
    warn: When `True` and an exception occurs, warn instead of raising the exception.
    hide: Controls the printing of the subprocess' stdout and stderr.

    Returns
    -------
    A `subprocess.CompletedProcess` object with the result of the asyncio.Process.

    Raises
    ------
    subprocess.CalledProcessError
        If an error occurs when running the command and `warn` is `False`.
    """

    logger.debug(f"Calling `asyncio.create_subprocess_exec` with {program_and_args=}")
    proc = await asyncio.create_subprocess_exec(
        *program_and_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        shell=False,
    )
    if input:
        logger.debug(f"Sending {input=!r} to the subprocess' stdin.")
    stdout, stderr = await proc.communicate(input.encode() if input else None)
    stdout = stdout.decode()
    stderr = stderr.decode()
    assert proc.returncode is not None
    result = subprocess.CompletedProcess(
        args=program_and_args,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    _warn_or_raise_on_error(
        result=result,
        input=input,
        warn=warn,
        hide=hide,
        _stack_level=_stack_level + 1,
    )
    _print_stdout_stderr(
        result,
        hide=hide,
        _stack_level=_stack_level + 1,
    )
    return result


def _warn_or_raise_on_error(
    result: subprocess.CompletedProcess,
    input: str | None,
    warn: bool,
    hide: Hide,
    _stack_level: int,
) -> None:
    """Common code between `run` and `run_async`.

    Note: This stuff is here to match the logging behaviour of `RemoteV1.run`.
    """
    if result.returncode == 0:
        return
    displayed_command = (
        result.args if isinstance(result.args, str) else shlex.join(result.args)
    )
    message = (
        f"Command {displayed_command!r}"
        + (f" with {input=!r}" if input else "")
        + f" exited with exit code {result.returncode}"
        + (f": {result.stderr}" if result.stderr else "")
    )

    logger.debug(message, stacklevel=_stack_level)
    if not warn:
        if result.stderr:
            logger.error(result.stderr, stacklevel=_stack_level)
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    if hide is not True:  # don't warn if hide is True.
        logger.warning(RuntimeWarning(message), stacklevel=_stack_level)


def _print_stdout_stderr(
    result: subprocess.CompletedProcess[str], hide: Hide, _stack_level: int = 2
):
    if result.stdout:
        logger.debug(f"stdout={result.stdout}", stacklevel=_stack_level)
        if hide not in [True, "out", "stdout"]:
            print(result.stdout)
    if result.stderr:
        logger.debug(f"stderr={result.stderr}", stacklevel=_stack_level)
        if hide not in [True, "err", "stderr"]:
            print(result.stderr, file=sys.stderr)
