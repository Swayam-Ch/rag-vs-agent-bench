import hashlib

import pytest

from ragbench.pipeline.checksums import read_checksums, sha256_file, verify, write_checksums
from ragbench.pipeline.fetch import check_corpus, main

# --------------------------------------------------------------------------- #
# checksums module
# --------------------------------------------------------------------------- #


def test_sha256_file_matches_hashlib(tmp_path):
    data = b"%PDF" + bytes(range(256)) * 1000  # bigger than one read block
    path = tmp_path / "a.pdf"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_round_trip_is_sorted_and_sha256sum_compatible(tmp_path):
    path = tmp_path / "checksums.sha256"
    entries = {"b.pdf": "b" * 64, "a.pdf": "a" * 64}
    write_checksums(path, entries)
    assert path.read_text() == f"{'a' * 64}  a.pdf\n{'b' * 64}  b.pdf\n"
    assert read_checksums(path) == entries


def test_read_missing_file_is_empty(tmp_path):
    assert read_checksums(tmp_path / "nope.sha256") == {}


def test_read_rejects_malformed_line(tmp_path):
    path = tmp_path / "checksums.sha256"
    path.write_text("not-a-hash a.pdf\n")
    with pytest.raises(ValueError, match="line 1"):
        read_checksums(path)


def test_verify_reports_missing_and_changed(tmp_path):
    (tmp_path / "ok.pdf").write_bytes(b"%PDF ok")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF tampered")
    expected = {
        "ok.pdf": hashlib.sha256(b"%PDF ok").hexdigest(),
        "bad.pdf": hashlib.sha256(b"%PDF original").hexdigest(),
        "gone.pdf": "0" * 64,
    }
    problems = verify(tmp_path, expected)
    assert len(problems) == 2
    assert any(p.startswith("changed: bad.pdf") for p in problems)
    assert "missing: gone.pdf" in problems


# --------------------------------------------------------------------------- #
# check_corpus: the lock-file behaviour
# --------------------------------------------------------------------------- #


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1706.03762.pdf").write_bytes(b"%PDF attention")
    (raw / "2005.11401.pdf").write_bytes(b"%PDF rag")
    return raw, tmp_path / "checksums.sha256"


def test_first_run_records_checksums(corpus):
    raw, sums = corpus
    problems, new = check_corpus(["1706.03762", "2005.11401"], raw, sums, record_new=True)
    assert problems == []
    assert new == ["1706.03762.pdf", "2005.11401.pdf"]
    assert set(read_checksums(sums)) == {"1706.03762.pdf", "2005.11401.pdf"}


def test_second_run_changes_nothing(corpus):
    raw, sums = corpus
    check_corpus(["1706.03762"], raw, sums, record_new=True)
    before = sums.read_text()
    problems, new = check_corpus(["1706.03762"], raw, sums, record_new=True)
    assert (problems, new) == ([], [])
    assert sums.read_text() == before


def test_one_changed_byte_is_caught(corpus):
    raw, sums = corpus
    check_corpus(["1706.03762"], raw, sums, record_new=True)
    pdf = raw / "1706.03762.pdf"
    pdf.write_bytes(pdf.read_bytes()[:-1] + b"X")
    problems, _ = check_corpus(["1706.03762"], raw, sums, record_new=True)
    assert problems and problems[0].startswith("changed: 1706.03762.pdf")


def test_verify_only_flags_unrecorded_papers(corpus):
    raw, sums = corpus
    problems, new = check_corpus(["1706.03762"], raw, sums, record_new=False)
    assert problems == ["no checksum recorded: 1706.03762.pdf"]
    assert new == []
    assert not sums.exists()  # verify mode never writes


def test_removed_sources_are_dropped_from_checksums(corpus):
    raw, sums = corpus
    check_corpus(["1706.03762", "2005.11401"], raw, sums, record_new=True)
    check_corpus(["1706.03762"], raw, sums, record_new=True)
    assert set(read_checksums(sums)) == {"1706.03762.pdf"}


# --------------------------------------------------------------------------- #
# CLI --verify (no network involved)
# --------------------------------------------------------------------------- #


def test_cli_verify_exit_codes(corpus, capsys):
    raw, sums = corpus
    sources = sums.parent / "sources.txt"
    sources.write_text("1706.03762\n")
    check_corpus(["1706.03762"], raw, sums, record_new=True)

    assert main([str(sources), str(raw), "--verify"]) == 0

    (raw / "1706.03762.pdf").write_bytes(b"%PDF different")
    assert main([str(sources), str(raw), "--verify"]) == 1
    assert "changed: 1706.03762.pdf" in capsys.readouterr().err
