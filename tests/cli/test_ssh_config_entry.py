""" Tests for the TypedDict for an entry in the `.ssh/config` file. """
import re
import subprocess
import textwrap

import pytest

from milatools.cli.ssh_config_entry import (
    SshConfigEntry,
    ssh_config_entry_keys_lowercase,
    to_entry,
)


@pytest.fixture(scope="module")
def man_ssh_config():
    pagename = "ssh_config"
    manpage = subprocess.check_output(["man", pagename], text=True)
    man_contents_per_key: dict[str, str] = {}
    # Parse the manpage contents.

    manpage_lines = str(manpage).splitlines()
    _first_line_indices = [
        index
        for index, line in enumerate(manpage_lines)
        if line.strip().startswith("Host")
    ]
    assert _first_line_indices
    first_line_index = _first_line_indices[0]

    argument_lines = manpage_lines[first_line_index:]

    def is_argument_description_start_line(line: str) -> bool:
        line = line.strip()
        starts_with_key = line.lower().startswith(
            tuple(ssh_config_entry_keys_lowercase)
        )
        if not starts_with_key:
            return False
        if not line:
            return False
        key, *rest = line.split()
        if not rest:
            return True
        rest_of_line = line[len(key) :]
        if rest_of_line.startswith("\t"):
            return True
        if rest[0][0].isupper():
            return True
        return False

    # For each argument in the manpage, store the description for that argument.
    arg_description_start_line_indices = [
        index
        for index, line in enumerate(argument_lines)
        if is_argument_description_start_line(line)
    ]
    arg_description_start_to_end = zip(
        arg_description_start_line_indices,
        arg_description_start_line_indices[1:] + [len(argument_lines)],
    )
    for start_index, end_index in arg_description_start_to_end:
        lines = argument_lines[start_index:end_index]
        key = lines[0].split()[0]
        lines[0] = " ".join(lines[0].split()[1:])

        content = textwrap.dedent("\n".join(lines))
        man_contents_per_key[key.lower()] = content
    return man_contents_per_key


keys_not_in_man_page = ["UsePrivilegedPort"]


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(
            k,
            marks=pytest.mark.xfail(
                reason=f"{wrong_k} doesn't show up in man page on my machine"
            ),
        )
        if k == wrong_k.lower()
        else k
        for k in ssh_config_entry_keys_lowercase
        for wrong_k in keys_not_in_man_page
    ],
)
def test_fields_match_man_contents(key: str, man_ssh_config: dict[str, str]):
    assert key.lower() in man_ssh_config.keys()
    # TODO: Check that the docstring for that field matches the description in the manpage.
    # there might be very tiny differences, like the quote symbols and the whitespace. But most of
    # the contents should match.
    # description = man_ssh_config[key.lower()]


@pytest.mark.parametrize(
    "dict, expected",
    [
        ({"Host": "bob.com"}, SshConfigEntry(Host="bob.com")),
        ({"host": "bob.com"}, SshConfigEntry(Host="bob.com")),
        ({"HOST": "bob.com"}, SshConfigEntry(Host="bob.com")),
    ],
)
def test_to_entry_valid(dict: dict, expected: SshConfigEntry):
    assert to_entry(dict) == expected


@pytest.mark.parametrize(
    "dict, error_type, error_message",
    [
        ({"Hosteoo": "bob.com"}, ValueError, "Invalid key: 'Hosteoo'"),
        (
            {"Hosteoo": "bob.com", "foo": 123},
            ValueError,
            "Invalid keys: ['Hosteoo', 'foo']",
        ),
        ({"host": "bob.com", "Host": "bobb.com"}, ValueError, "Key collision"),
    ],
)
def test_to_entry_invalid(dict: dict, error_type: type[Exception], error_message: str):
    match = re.escape(error_message)
    assert isinstance(match, str)
    with pytest.raises(error_type, match=match):
        to_entry(dict)
