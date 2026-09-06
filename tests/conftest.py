"""Shared fixtures.

pytest's tmp_path fixture is unusable under the DSH file sandbox:
tempfile.mkdtemp creates mode-0700 dirs whose descendants become
unwritable by the creating process itself. We instead create plain-mode
(Path.mkdir default) dirs under the repo root, which are fully usable.
"""

import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workdir():
    """A writable, best-effort-cleaned temp dir inside the repo root."""
    d = REPO_ROOT / f".testwork-{uuid.uuid4().hex[:8]}"
    d.mkdir()
    yield d
    # Best-effort cleanup: files first, then dirs, deepest first.
    try:
        for child in sorted(d.rglob("*"), key=lambda p: -len(p.parts)):
            if child.is_file():
                child.unlink()
    except OSError:
        pass
    try:
        for child in sorted(d.rglob("*"), key=lambda p: -len(p.parts)):
            if child.is_dir() and child != d:
                child.rmdir()
        d.rmdir()
    except OSError:
        pass