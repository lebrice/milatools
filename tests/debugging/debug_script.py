import fabric

from milatools.cli.commands import setup_logging
from milatools.cli.remote import Remote

setup_logging(verbose=2)
fabric_output = (
    fabric.Connection(
        "mila",
        connect_kwargs={"banner_timeout": 60},
    )
    .run("echo OK")
    .stdout
)

print(f"{fabric_output=}")
print(f"{Remote('mila').get_output('echo OK')=}")
