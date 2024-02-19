from __future__ import annotations

import datetime
import functools
import os
import time
from logging import getLogger as get_logger

import pytest
from fabric.connection import Connection

from milatools.cli.remote import Remote
from milatools.cli.utils import cluster_to_connect_kwargs
from tests.cli.common import in_github_CI

logger = get_logger(__name__)
JOB_NAME = "milatools_test"
WCKEY = "milatools_test"

SLURM_CLUSTER = os.environ.get("SLURM_CLUSTER", "mila" if not in_github_CI else None)
"""The name of the slurm cluster to use for tests.

When running the tests on a dev machine, this defaults to the Mila cluster. Set to
`None` when running on the github CI.
"""

MAX_JOB_DURATION = datetime.timedelta(seconds=10)

hangs_in_github_CI = pytest.mark.skipif(
    SLURM_CLUSTER == "localhost",
    reason=(
        "TODO: Hangs in the GitHub CI (probably because it runs salloc or sbatch on a "
        "cluster with only `localhost` as a 'compute' node?)"
    ),
)


@pytest.fixture(scope="session", autouse=True)
def cancel_all_milatools_jobs_before_and_after_tests(cluster: str):
    # Note: need to recreate this because login_node is a function-scoped fixture.
    login_node = Remote(cluster)
    logger.info(
        f"Cancelling milatools test jobs on {cluster} before running integration tests."
    )
    login_node.run(f"scancel -u $USER --wckey={WCKEY}")
    time.sleep(1)
    yield
    logger.info(
        f"Cancelling milatools test jobs on {cluster} after running integration tests."
    )
    login_node.run(f"scancel -u $USER --wckey={WCKEY}")
    time.sleep(1)
    # Display the output of squeue just to be sure that the jobs were cancelled.
    logger.info(f"Checking that all jobs have been cancelked on {cluster}...")
    login_node._run("squeue --me", echo=True, in_stream=False)


@functools.lru_cache
def get_slurm_account(cluster: str) -> str:
    """Gets the SLURM account of the user using sacctmgr on the slurm cluster.

    When there are multiple accounts, this selects the first account, alphabetically.

    On DRAC cluster, this uses the `def` allocations instead of `rrg`, and when
    the rest of the accounts are the same up to a '_cpu' or '_gpu' suffix, it uses
    '_cpu'.

    For example:

    ```text
    def-someprofessor_cpu  <-- this one is used.
    def-someprofessor_gpu
    rrg-someprofessor_cpu
    rrg-someprofessor_gpu
    ```
    """
    logger.info(
        f"Fetching the list of SLURM accounts available on the {cluster} cluster."
    )
    result = Remote(cluster).run(
        "sacctmgr --noheader show associations where user=$USER format=Account%50"
    )
    accounts = [line.strip() for line in result.stdout.splitlines()]
    assert accounts
    logger.info(f"Accounts on the slurm cluster {cluster}: {accounts}")
    account = sorted(accounts)[0]
    logger.info(f"Using account {account} to launch jobs in tests.")
    return account


@pytest.fixture()
def allocation_flags(cluster: str, request: pytest.FixtureRequest) -> list[str]:
    # note: thanks to lru_cache, this is only making one ssh connection per cluster.
    account = get_slurm_account(cluster)
    allocation_options = {
        "job-name": JOB_NAME,
        "wckey": WCKEY,
        "account": account,
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": 1,
        "mem": "1G",
        "time": MAX_JOB_DURATION,
        "oversubscribe": None,  # allow multiple such jobs to share resources.
    }
    overrides = getattr(request, "param", {})
    assert isinstance(overrides, dict)
    if overrides:
        print(f"Overriding allocation options with {overrides}")
        allocation_options.update(overrides)
    return [
        f"--{key}={value}" if value is not None else f"--{key}"
        for key, value in allocation_options.items()
    ]


@pytest.fixture(scope="session", params=[SLURM_CLUSTER])
def cluster(
    request: pytest.FixtureRequest,
) -> str:
    """Fixture that gives the hostname of the slurm cluster to use for tests.

    NOTE: The `cluster` can also be parametrized indirectly by tests, for example:

    ```python
    @pytest.mark.parametrize("cluster", ["mila", "some_cluster"], indirect=True)
    def test_something(remote: Remote):
        ...  # here the remote is connected to the cluster specified above!
    ```
    """
    slurm_cluster_hostname = request.param

    if not slurm_cluster_hostname:
        pytest.skip("Requires ssh access to a SLURM cluster.")
    return slurm_cluster_hostname


@pytest.fixture(scope="function")
def login_node(cluster: str) -> Remote:
    """Fixture that gives a Remote connected to the login node of a slurm cluster.

    NOTE: Making this a function-scoped fixture because the Connection object of the
    Remote seems to be passed (and reused?) when creating the `SlurmRemote` object.

    We want to avoid that, because `SlurmRemote` creates jobs when it runs commands.
    We also don't want to accidentally end up with `login_node` that runs commands on
    compute nodes because a previous test kept the same connection object while doing
    salloc (just in case that were to happen).
    """

    return Remote(
        cluster,
        connection=Connection(
            cluster, connect_kwargs=cluster_to_connect_kwargs.get(cluster)
        ),
    )
