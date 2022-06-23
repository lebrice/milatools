""" IDEA: Create a command that sets up a user cache directory for commonly used libraries.

This can be used to avoid having to download files to the $HOME directory, as well as remove
duplicated files (e.g. the same file in the shared cache dir and the user cache dir).

The user cache directory should be writeable, and will contain some read-only links to the
files contained in the "shared" cache directory, (e.g. managed by the IT/IDT Team at Mila).

Also sets the environment variables so that this new cache location is used by default by those
libraries.
"""
from __future__ import annotations
from dataclasses import dataclass

import logging
import shutil
from typing import Literal
from pathlib import Path
import os

from tqdm import tqdm
from logging import getLogger as get_logger
from dataclasses import dataclass
from simple_parsing import ArgumentParser, choice

logger = get_logger(__name__)

SupportedFramework = Literal["huggingface", "torch", "ALL"]

SCRATCH = Path.home() / "scratch"
DEFAULT_USER_CACHE_DIR = SCRATCH / ".cache"
# TODO: Change to an actual IDT-approved, read-only directory. Using another dir in my scratch for now.
# SHARED_CACHE_DIR = Path("/network/shared_cache")
SHARED_CACHE_DIR = SCRATCH / "shared_cache"


# IDEA: Could list the files in the shared cache dir, to show the available frameworks!
# FIXME: When specifying the shared_cache_dir with an argument, setting 'choices' isn't correct,
# because the actual shared cache_dir that is pointed to by `shared_cache_dir` might have
# different files than the default shared cache_dir.
subdirectories = [p.name for p in SHARED_CACHE_DIR.iterdir()]


@dataclass
class Options:
    """ Options for the setup_cache command. """

    user_cache_dir: Path = DEFAULT_USER_CACHE_DIR
    """The user cache directory. Should probably be in $SCRATCH (not $HOME!) """

    shared_cache_dir: Path = SHARED_CACHE_DIR
    """ The path to the shared cache directory.
    
    This defaults to the path of the shared cache setup by the IDT team on the Mila cluster.
    """

    framework_subdirectory: str = choice(subdirectories + ["all"], default="all")
    """The name of a subdirectory of `shared_cache_dir` to link. By default, creates symlinks for
    every file in the `shared_cache_dir` directory.
    """

    def __post_init__(self):
        if self.framework_subdirectory != "all":
            subdirectories = [p.name for p in self.shared_cache_dir.iterdir()]
            if self.framework_subdirectory not in subdirectories:
                raise ValueError(
                    f"The framework subdirectory '{self.framework_subdirectory}' does not exist in "
                    f"{self.shared_cache_dir}. \n"
                    f"Frameworks/subdirectories available in the shared cache: {subdirectories}"
                )

            self.user_cache_dir = self.user_cache_dir / self.framework_subdirectory
            self.shared_cache_dir = self.shared_cache_dir / self.framework_subdirectory


def setup_cache(user_cache_dir: Path, shared_cache_dir: Path,) -> None:
    """Set up the user cache directory. 

    1. If the `user_cache_dir` directory doesn't exist, creates it.
    2. Removes broken symlinks in the user cache directory if they point to files in
       `shared_cache_dir` that don't exist anymore.
    3. For every file in the shared cache dir, creates a (symbolic?) link to it in the
       `user_cache_dir`.
    """

    if not user_cache_dir.exists():
        user_cache_dir.mkdir(parents=True, exist_ok=False)
    if not user_cache_dir.is_dir():
        raise RuntimeError(f"cache_dir is not a directory: {user_cache_dir}")
    if not shared_cache_dir.is_dir():
        raise RuntimeError(
            f"The shared cache directory {shared_cache_dir} doesn't exist, or isn't a directory! "
        )

    delete_broken_symlinks_to_shared_cache(user_cache_dir, shared_cache_dir)
    create_links(user_cache_dir, shared_cache_dir)

    set_environment_variables(user_cache_dir)


def set_environment_variables(user_cache_dir: Path):
    """Set the relevant environment variables for each library so they start to use the new cache
    dir.
    """
    # TODO: These changes won't persist. We probably need to add a block of code in .bashrc

    os.environ["TORCH_HOME"] = str(user_cache_dir / "torch")
    os.environ["HF_HOME"] = str(user_cache_dir / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(
        user_cache_dir / "huggingface" / "transformers"
    )


def is_child(path: Path, parent: Path) -> bool:
    """Return True if the path is under the parent directory."""
    if path == parent:
        return False
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def delete_broken_symlinks_to_shared_cache(
    user_cache_dir: Path, shared_cache_dir: Path
):
    """Delete all symlinks in the user cache directory that point to files that don't exist anymore
    in the shared cache directory. """
    for file in user_cache_dir.rglob("*"):
        if file.is_symlink():
            target = file.resolve()
            if is_child(target, shared_cache_dir) and not target.exists():
                logger.debug(f"Removing broken symlink: {file}")
                if file.is_dir():
                    file.rmdir()
                else:
                    file.unlink()


def create_links(user_cache_dir: Path, shared_cache_dir: Path):
    """Create symlinks to the shared cache directory in the user cache directory. """
    # For every file in the shared cache dir, create a (symbolic?) link to it in the user cache dir
    pbar = tqdm()

    def _copy_fn(src: str, dst: str) -> None:
        # NOTE: This also overwrites the files in the user directory with symlinks to the same files in
        # the shared directory. We might not necessarily want to do that.
        # For instance, we might want to do a checksum or something first, to check that they have
        # exactly the same contents.
        src_path = Path(src)
        dst_path = Path(dst)
        rel_d = dst_path.relative_to(user_cache_dir)
        rel_s = src_path.relative_to(shared_cache_dir)

        if dst_path.exists():
            if dst_path.is_symlink():
                # From a previous run.
                return
            # Replace "real" files with symlinks.
            dst_path.unlink()

        # print(f"Linking {rel_s}")
        pbar.set_description(f"Linking {rel_s}")
        pbar.update(1)
        os.symlink(src, dst)  # Create symlinks instead of copying.

    shutil.copytree(
        shared_cache_dir,
        user_cache_dir,
        symlinks=True,
        copy_function=_copy_fn,
        dirs_exist_ok=True,
    )


def main():

    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()

    options: Options = args.options
    setup_cache(options.user_cache_dir, options.shared_cache_dir)


if __name__ == "__main__":
    main()
