import inspect
import shlex
import subprocess
import sys
import unittest
import unittest.mock

import pytest

import milatools
import milatools.cli.commands
from milatools.cli.commands import lab, main
from milatools.cli.remote import Remote


def test_mila_serve_lab_cli_to_fn_arguments(monkeypatch: pytest.MonkeyPatch):
    """Test that the command on the CLI produces the expected function arguments."""
    # IDEA: This is to reproduce an issue that arises with the following command:
    # mila serve lab --alloc --time=4:00:00 --gres gpu:1 --cpus-per-task=4 \
    #   --mem-per-cpu=4GB
    # In order to write a test for it, we first need to make sure that we know which
    # arguments are passed to the `lab` function when this command is run.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mila",
            "serve",
            "lab",
            "--alloc",
            "--time=4:00:00",
            "--gres",
            "gpu:1",
            "--cpus-per-task=4",
            "--mem-per-cpu=4GB",
        ],
    )
    mock_lab = unittest.mock.Mock(spec=lab)
    monkeypatch.setattr(milatools.cli.commands, lab.__name__, mock_lab)

    main(
        shlex.split(
            "serve lab --alloc --time=4:00:00 --gres gpu:1 "
            "--cpus-per-task=4 --mem-per-cpu=4GB"
        )
    )

    mock_lab.assert_called_once_with(
        path=None,
        alloc=[
            "--time=4:00:00",
            "--gres",
            "gpu:1",
            "--cpus-per-task=4",
            "--mem-per-cpu=4GB",
        ],
        job=None,
        name=None,
        node=None,
        persist=False,
        port=None,
        profile=None,
    )


@pytest.fixture(scope="session")
def milatools_test_profile(cluster: str):
    login_node = Remote(cluster)
    env_name = profile_name = "milatools-test"
    # TODO: terribly stupid. Can't seem to run a few commands in a row within the same
    # connection. Makes no sense to me.
    if env_name not in login_node.get_output(
        " && ".join(
            [
                "module --quiet purge",
                "module load anaconda/3",
                "conda env list",
            ]
        )
    ):
        login_node.run(
            " && ".join(
                [
                    "module --quiet purge",
                    "module load anaconda/3",
                    (
                        f"srun --mem=4G --cpus-per-task=2 --time=00:03:00 "
                        f"conda create --name {env_name} python=3.11 jupyterlab -y"
                    ),
                ]
            )
        )
    login_node.puttext(
        "\n".join(
            [
                "module load anaconda/3",
                "conda activate ~/.conda/milatools-test",
            ]
        ),
        f".milatools/profiles/{profile_name}.bash",
    )
    return "milatools-test"


def test_mila_serve_lab_error(milatools_test_profile: str):
    lab(
        path=None,
        alloc=[
            "--time=4:00:00",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem-per-cpu=4GB",
        ],
        job=None,
        node=None,
        name=None,
        persist=False,
        port=None,
        profile=milatools_test_profile,
    )
