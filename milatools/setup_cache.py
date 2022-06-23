""" IDEA: Create a command that sets up a cache directory on SCRATCH for commonly used libraries,
e.g. HuggingFace, torchvision, etc.

- Also set the environment variables so that this new cache location is used by default by those libraries.
- The user-specific cache directory should be writeable, and contain some read-only links to the
  files contained in the "shared" cache directory, managed by the IT/IDT Team at Mila. 
"""


import shutil
from typing import Literal
from pathlib import Path
import os

from tqdm import tqdm
from logging import getLogger as get_logger

logger = get_logger(__name__)
SupportedFramework = Literal["transformers", "torchvision"]

SCRATCH = Path.home() / "scratch"
DEFAULT_USER_CACHE_DIR = SCRATCH / ".cache"
# TODO: Change to an actual directory. Using another dir in my scratch for now.
# SHARED_CACHE_DIR = Path("/network/shared_cache")
SHARED_CACHE_DIR = SCRATCH / "shared_cache"


def setup_cache(
    user_cache_dir: Path,
    framework: SupportedFramework,
    shared_cache_dir: Path = SHARED_CACHE_DIR,
) -> None:
    """Set up the cache directory for the given framework. """

    if framework == "transformers":
        setup_huggingface_cache(user_cache_dir)
    elif framework == "torchvision":
        user_torchvision_dir = user_cache_dir / "torch"
        shared_torchvision_dir = shared_cache_dir / "torch"

        setup_torchvision_cache(user_torchvision_dir, shared_torchvision_dir)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def setup_huggingface_cache(cache_dir: Path):
    raise NotImplementedError("Setup of huggingface cache is not yet implemented.")


def setup_torchvision_cache(
    user_cache_dir: Path = SCRATCH / "torch",
    shared_cache_dir: Path = SHARED_CACHE_DIR / "torch",
):
    """ TODO: Setup the torchvision cache.
    
    1. If the user cache dir doesn't exist, create it.
    2. Delete all symlinks in the user cache directory that point to files that don't exist anymore
       in the shared cache directory.
    3. For every file in the shared cache dir, create a (symbolic?) link to it in the user cache
       dir.
    """
    if not user_cache_dir.exists():
        user_cache_dir.mkdir(parents=True, exist_ok=False)
    if not user_cache_dir.is_dir():
        raise RuntimeError(f"cache_dir is not a directory: {user_cache_dir}")
    if not shared_cache_dir.is_dir():
        raise RuntimeError(
            f"The shared cache directory {shared_cache_dir} doesn't exist, or isn't a directory! "
        )

    if "TORCH_HOME" not in os.environ:
        # TODO: These changes won't persist. We probably need to add a block of code in .bashrc
        os.environ["TORCH_HOME"] = str(user_cache_dir)

    delete_broken_symlinks(user_cache_dir)

    create_links(user_cache_dir, shared_cache_dir)


def is_child(path: Path, parent: Path) -> bool:
    """Return True if the path is under the parent directory."""
    if path == parent:
        return False
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def delete_broken_symlinks(user_cache_dir: Path):
    """Delete all symlinks in the user cache directory that point to files that don't exist anymore
    in the shared cache directory. """
    for file in user_cache_dir.rglob("*"):
        if file.is_symlink():
            if not file.resolve().exists():
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

    # for shared_path in shared_cache_dir.rglob("*"):
    #     shared_path_relative = shared_path.relative_to(shared_cache_dir)
    #     user_path = user_cache_dir / shared_path_relative

    #     if not user_path.exists():
    #         # Create a link to the shared version of the file.
    #         user_path.symlink_to(shared_path, target_is_directory=shared_path.is_dir())
    #         continue

    #     # The file exists.

    #     if user_path.is_symlink():
    #         # The path is a symlink that points to an existing file/dir in the shared cache dir.
    #         # (This is true since we purged the dangling symlinks above.)
    #         continue

    #     # The path is a real file / directory in the user's cache_dir.

    #     # NOTE: While we *could* just go to the next iteration here, since we've done what we
    #     # wanted to do, it might be a good idea to replace dupliate "real" files in the user cache
    #     # with symlinks to the same files in the shared cache.
    #     # Assuming that the datasets / model weights are stored on SCRATCH, I can't think
    #     # of any reason NOT to do this, since it's just as fast, but will make them use less
    #     # space.
    #     if user_path.is_file():
    #         user_path.symlink_to(shared_path)

    #     elif user_path.is_dir():
    #         # There is a "real" directory in the user's cache, and we have a directory in the
    #         # shared cache with the same name. It's important to not just replace the user's dir
    #         # with a link to the shared dir, since the user's dir may contain files that the shared
    #         # dir doesn't have!
    #         # We can probably do nothing here, since we're going to recurse into the other files in
    #         # that directory anyway.
    #         pass


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("framework", type=str, default="torchvision")
    parser.add_argument("--user_cache", type=Path, default=DEFAULT_USER_CACHE_DIR)
    parser.add_argument("--shared_cache", type=Path, default=SHARED_CACHE_DIR)
    args = parser.parse_args()
    framework = args.framework
    user_cache_dir = args.user_cache
    shared_cache_dir = args.shared_cache
    setup_cache(user_cache_dir, framework, shared_cache_dir)


if __name__ == "__main__":
    main()
