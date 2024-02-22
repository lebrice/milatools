import fabric

from milatools.cli.commands import setup_logging
from milatools.cli.remote import Remote


def main():
    host = "mila"
    command = "echo OK"
    setup_logging(verbose=2)
    fabric_output = (
        fabric.Connection(
            host,
            connect_kwargs={"banner_timeout": 60},
        )
        .run(command)
        .stdout.strip()
    )
    print(f"{fabric_output=}")
    print(f"{Remote(host).get_output(command)=}")


if __name__ == "__main__":
    main()
