from __future__ import annotations

import contextvars
import itertools
import random
import shlex
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import overload

import blessed
import paramiko
import questionary as qn
from invoke.exceptions import UnexpectedExit
from sshconf import read_ssh_config
from typing_extensions import Unpack

from .ssh_config_entry import SshConfigEntry, SshConfigEntryLowercase, to_entry

control_file_var = contextvars.ContextVar("control_file", default="/dev/null")

T = blessed.Terminal()

here = Path(__file__).parent

style = qn.Style(
    [
        ("envname", "yellow bold"),
        ("envpath", "cyan"),
        ("prefix", "bold"),
        ("special", "orange bold"),
        ("cancel", "grey bold"),
    ]
)

vowels = list("aeiou")
consonants = list("bdfgjklmnprstvz")
syllables = [
    "".join(letters) for letters in itertools.product(consonants, vowels)
]


def randname():
    a = random.choice(syllables)
    b = random.choice(syllables)
    c = random.choice(syllables)
    d = random.choice(syllables)
    return f"{a}{b}-{c}{d}"


@contextmanager
def with_control_file(remote, name=None):
    name = name or randname()
    pth = f".milatools/control/{name}"
    remote.run("mkdir -p ~/.milatools/control", hide=True)

    try:
        remote.simple_run(f"[ -f {pth} ]")
        exit(
            f"Server {name} already exists. You may use mila serve kill to remove it."
        )
    except UnexpectedExit:
        pass

    token = control_file_var.set(pth)
    try:
        yield pth
    finally:
        control_file_var.reset(token)


class MilatoolsUserError(Exception):
    pass


class CommandNotFoundError(MilatoolsUserError):
    # Instructions to install certain commands if they are not found
    instructions = {
        "code": (
            "To fix this, try starting VSCode, then hit Cmd+Shift+P,"
            " type 'install code command' in the box, and hit Enter."
            " You might need to restart your shell."
        )
    }

    def __init__(self, command):
        super().__init__(command)

    def __str__(self):
        cmd = self.args[0]
        message = f"Command '{cmd}' does not exist locally."
        supp = self.instructions.get(cmd, None)
        if supp:
            message += f" {supp}"
        return message


class SSHConnectionError(paramiko.SSHException):
    def __init__(self, node_hostname: str, error: paramiko.SSHException):
        super().__init__()
        self.node_hostname = node_hostname
        self.error = error

    def __str__(self):
        return (
            "An error happened while trying to establish a connection with {0}".format(
                self.node_hostname
            )
            + "\n\t"
            + "-The cluster might be under maintenance"
            + "\n\t   "
            + "Check #mila-cluster for updates on the state of the cluster"
            + "\n\t"
            + "-Check the status of your connection to the cluster by ssh'ing onto it."
            + "\n\t"
            + "-Retry connecting with mila"
            + "\n\t"
            + "-Try to exclude the node with -x {0} parameter".format(
                self.node_hostname
            )
        )


def yn(prompt: str, default: bool = True) -> bool:
    return qn.confirm(prompt, default=default).unsafe_ask()


def askpath(prompt, remote):
    while True:
        pth = qn.text(prompt).unsafe_ask()
        try:
            remote.simple_run(f"[ -d {pth} ]")
        except UnexpectedExit:
            qn.print(f"Path {pth} does not exist")
            continue
        return pth


# This is the implementation of shlex.join in Python >= 3.8
def shjoin(split_command):
    """Return a shell-escaped string from *split_command*."""
    return " ".join(shlex.quote(arg) for arg in split_command)


class SSHConfig:
    """Wrapper around sshconf with some extra niceties."""

    def __init__(self, path: str | Path):
        self.cfg = read_ssh_config(path)
        self.remove = self.cfg.remove
        self.rename = self.cfg.rename
        self.save = self.cfg.save
        self.hosts = self.cfg.hosts

    @overload
    def add(
        self,
        host: str,
        Host: str | None = None,
        **kwargs: Unpack[SshConfigEntry],
    ) -> SshConfigEntry:
        ...

    @overload
    def add(self, **kwargs: Unpack[SshConfigEntry]) -> SshConfigEntry:
        ...

    @overload
    def add(self, **kwargs: Unpack[SshConfigEntryLowercase]) -> SshConfigEntry:
        ...

    def add(
        self, host: str | None = None, Host: str | None = None, **kwargs
    ) -> SshConfigEntry:
        """
        Add an entry for the given host to the SSH configuration.

        Parameters
        ----------
        host: The Host entry to add.
        **kwargs: The parameters for the host (without "Host" parameter itself)

        Returns
        -------
        The new ssh_config entry. This is a dictionary, whose keys and value types are annotated in
        the `SshConfigEntry` TypedDict.

        Raises a ValueError if there are invalid keys in kwargs.
        """
        assert not (host and Host)
        host = Host or host
        # NOTE: transforms the keys to match their CamelCase entries in the man page. Also raises a
        # ValueError if the key is not a valid entry.
        entry = to_entry(**kwargs)
        self.cfg.add(host, **entry)
        return entry

    def host(self, host: str) -> SshConfigEntryLowercase:
        return SshConfigEntryLowercase(**self.cfg.host(host))

    def hoststring(self, host: str):
        lines = []
        for filename, cfg in self.cfg.configs_:
            lines += [line.line for line in cfg.lines_ if line.host == host]
        return "\n".join(lines)

    def confirm(self, host: str):
        print(
            T.bold(
                "The following code will be appended to your ~/.ssh/config:\n"
            )
        )
        print(self.hoststring(host))
        return yn("\nIs this OK?")


def qualified(node_name: str) -> str:
    """Return the fully qualified name corresponding to this node name."""

    if "." not in node_name and not node_name.endswith(".server.mila.quebec"):
        node_name = f"{node_name}.server.mila.quebec"
    return node_name


def get_fully_qualified_name() -> str:
    """Return the fully qualified name of the current machine.

    Much faster than socket.getfqdn() on Mac. Falls back to that if the hostname command is not available.
    """
    try:
        return (
            subprocess.check_output(["hostname", "-f"]).decode("utf-8").strip()
        )
    except Exception:
        # Fall back, e.g. on Windows.
        return socket.getfqdn()
