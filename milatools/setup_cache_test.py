from __future__ import annotations
import os
from pathlib import Path
from typing import Callable
import pytest
import torchvision
from torchvision.models import resnet18

from milatools.setup_cache import create_links


def create_dummy_dir_tree(
    parent_dir: Path, files: list[str], mode: int | None = None
) -> None:
    parent_dir.mkdir()
    paths = [parent_dir / file for file in files]
    for path in paths:
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(f"Hello, this is the content of {path}")
        if mode is not None:
            path.chmod(mode)


def test_create_links(tmp_path: Path):
    # Create /foo/bar/baz.txt, call setup_cache, and see if it works.
    user_dir = tmp_path / "user"
    shared_dir = tmp_path / "shared"

    user_files = ["foo.txt", "bar/baz.txt"]
    shared_files = ["shared.txt"]

    # Create some dummy directories.
    create_dummy_dir_tree(user_dir, user_files)
    create_dummy_dir_tree(shared_dir, shared_files)

    user_paths = [user_dir / p for p in user_files]
    shared_paths = [shared_dir / p for p in shared_files]
    user_file_contents = {p: p.read_text() for p in user_paths}
    shared_file_contents = {p: p.read_text() for p in shared_paths}

    create_links(user_dir, shared_dir)

    for user_file in user_files:
        user_path = user_dir / user_file
        assert user_path.exists()
        if user_file not in shared_files:
            # User paths and contents stay the same.
            assert not user_path.is_symlink()
            assert user_path.read_text() == user_file_contents[user_path]
        else:
            # The 'real' path should be replaced with a symbolic link!
            assert user_path.is_symlink()
            assert user_path.read_text() == shared_file_contents[user_path]

    for shared_file in shared_files:
        new_symlink_location = user_dir / shared_file
        assert new_symlink_location.exists()
        target = shared_dir / shared_file
        assert new_symlink_location.is_symlink()
        assert new_symlink_location.resolve() == target
        assert new_symlink_location.read_text() == shared_file_contents[target]


@pytest.fixture
def empty_shared_cache_dir(tmp_path_factory, monkeypatch):
    """ Fake shared cache directory. """
    shared_cache_dir: Path = tmp_path_factory.mktemp("shared_empty_cache_dir")
    shared_cache_dir.chmod(0o444)

    monkeypatch.setitem(os.environ, "TORCH_HOME", str(shared_cache_dir))
    yield shared_cache_dir


@pytest.fixture(scope="session")
def polulated_shared_cache_dir(tmp_path_factory, monkeypatch):
    shared_cache_dir: Path = tmp_path_factory.mktemp("shared_cache_dir")
    monkeypatch.setitem(os.environ, "TORCH_HOME", str(shared_cache_dir.absolute()))
    resnet18(pretrained=True, progress=True)
    checkpoint_file = shared_cache_dir / "hub" / "checkpoints" / "resnet18-f37072fd.pth"

    assert checkpoint_file.exists()
    # TODO: Double-check that this works.
    mode = checkpoint_file.stat()
    shared_cache_dir_mode = shared_cache_dir.stat()
    checkpoint_file.chmod(0o444)
    shared_cache_dir.chmod(0o444)
    monkeypatch.setitem(os.environ, "TORCH_HOME", str(shared_cache_dir))
    yield
    # NOTE: This might not actually be necessary. My concern was that pytest wouldn't be able to
    # delete these if I changed the permission here.
    checkpoint_file.chmod(mode.st_mode)
    shared_cache_dir.chmod(shared_cache_dir_mode.st_mode)


library_function_to_created_files = {
    resnet18: [Path("hub/checkpoints/resnet18-f37072fd.pth")],
}


def get_all_files_in_dir(dir_path: Path) -> list[Path]:
    return [p.relative_to(dir_path) for p in dir_path.glob("**/*") if not p.is_dir()]


class TestTorchvision:
    """ TODO: Add more tests specific to each library. """

    def test_cant_write_to_shared_cache_dir(
        self, empty_shared_cache_dir: Path, monkeypatch
    ):
        monkeypatch.setitem(os.environ, "TORCH_HOME", str(empty_shared_cache_dir))
        assert len(list(empty_shared_cache_dir.iterdir())) == 0

        with pytest.raises(IOError, match="Permission denied"):
            resnet18(pretrained=True, progress=True)

    def test_changing_torch_home_works(self, tmp_path: Path, monkeypatch):
        """Test that changing the TORCH_HOME environment variable changes where the weights are saved. """
        monkeypatch.setitem(os.environ, "TORCH_HOME", str(tmp_path.absolute()))
        assert len(list(tmp_path.iterdir())) == 0

        resnet18(pretrained=True, progress=True)

        files_after = list(tmp_path.iterdir())
        assert len(files_after) == 1

        # Basically, we want to check that this created a `hub/`
        assert get_all_files_in_dir(tmp_path) == [
            Path("hub") / "checkpoints" / "resnet18-f37072fd.pth"
        ]
