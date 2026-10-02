"""SHA-256 checksums for the downloaded corpus, stored like a lock file.

`data/checksums.sha256` is committed to git. The first time a paper is downloaded its
hash is recorded; on every later run the file on disk must match that hash exactly.
That's what lets anyone prove they benchmarked the very same PDFs.

The file uses the standard `sha256sum` format (`<hash>  <file name>`), so it can also
be checked without this project:

    cd data/raw && shasum -a 256 -c ../checksums.sha256
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_BLOCK = 1 << 16  # read 64 KB at a time so large files never sit in memory whole


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, reading it in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """Parse a checksums file into {file name: hash}. A missing file means no entries."""
    if not path.exists():
        return {}
    entries: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64:
            raise ValueError(f"{path.name} line {lineno}: expected '<sha256>  <file name>'")
        entries[name.strip()] = digest.lower()
    return entries


def write_checksums(path: Path, entries: dict[str, str]) -> None:
    """Write {file name: hash} sorted by name, so the file diffs cleanly in git."""
    lines = [f"{digest}  {name}\n" for name, digest in sorted(entries.items())]
    path.write_text("".join(lines), encoding="utf-8")


def verify(files_dir: Path, expected: dict[str, str]) -> list[str]:
    """Compare files on disk with expected hashes. Returns a list of problems (empty = OK)."""
    problems: list[str] = []
    for name, digest in sorted(expected.items()):
        path = files_dir / name
        if not path.exists():
            problems.append(f"missing: {name}")
        elif (actual := sha256_file(path)) != digest:
            problems.append(f"changed: {name} (expected {digest[:12]}…, got {actual[:12]}…)")
    return problems
