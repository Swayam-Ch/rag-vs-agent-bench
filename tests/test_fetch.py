import pytest

from ragbench.pipeline import fetch
from ragbench.pipeline.fetch import download_pdf, fetch_all, main, parse_sources

# --------------------------------------------------------------------------- #
# parse_sources: pure logic, no network
# --------------------------------------------------------------------------- #


def test_parse_ignores_comments_and_blanks():
    text = "# header\n\n2005.11401  # RAG\n   \n1706.03762\n"
    assert parse_sources(text) == ["2005.11401", "1706.03762"]


def test_parse_keeps_version_suffix():
    assert parse_sources("2005.11401v4") == ["2005.11401v4"]


def test_parse_dedupes_in_order():
    assert parse_sources("1706.03762\n2005.11401\n1706.03762") == ["1706.03762", "2005.11401"]


@pytest.mark.parametrize("bad", ["not-an-id", "1706.037", "hep-th/9901001", "1706.03762 extra"])
def test_parse_rejects_bad_id_with_line_number(bad):
    with pytest.raises(ValueError, match="line 2"):
        parse_sources(f"1706.03762\n{bad}\n")


def test_committed_sources_file_is_valid():
    """The real corpus definition must always parse."""
    from pathlib import Path

    sources = Path(__file__).parent.parent / "data" / "sources.txt"
    ids = parse_sources(sources.read_text())
    assert len(ids) >= 20


# --------------------------------------------------------------------------- #
# download: a fake session instead of the internet
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, content: bytes = b"%PDF-1.5 fake", status: int = 200):
        self.content, self.status = content, status
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        return FakeResponse(self.content, self.status)


def test_download_writes_pdf(tmp_path):
    session = FakeSession()
    path = download_pdf("2005.11401", tmp_path, session)
    assert path == tmp_path / "2005.11401.pdf"
    assert path.read_bytes().startswith(b"%PDF")
    assert session.requested == ["https://arxiv.org/pdf/2005.11401"]
    assert not list(tmp_path.glob("*.part"))  # temp file renamed away


def test_download_rejects_non_pdf(tmp_path):
    with pytest.raises(ValueError, match="not a PDF"):
        download_pdf("2005.11401", tmp_path, FakeSession(content=b"<html>slow down</html>"))
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_download_raises_on_http_error(tmp_path):
    with pytest.raises(RuntimeError, match="404"):
        download_pdf("2005.11401", tmp_path, FakeSession(status=404))


def test_fetch_all_skips_existing(tmp_path):
    (tmp_path / "1706.03762.pdf").write_bytes(b"%PDF already here")
    session = FakeSession()
    downloaded, skipped = fetch_all(["1706.03762", "2005.11401"], tmp_path, session, delay=0)
    assert skipped == ["1706.03762"]
    assert downloaded == [tmp_path / "2005.11401.pdf"]
    assert len(session.requested) == 1


def test_fetch_all_sleeps_only_between_downloads(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    fetch_all(["2005.11401", "1706.03762", "1810.04805"], tmp_path, FakeSession(), delay=3)
    assert sleeps == [3, 3]  # 3 downloads -> 2 pauses, none before the first


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_reports_bad_sources_file(tmp_path, capsys):
    sources = tmp_path / "sources.txt"
    sources.write_text("garbage\n")
    assert main([str(sources), str(tmp_path / "out")]) == 1
    assert "line 1" in capsys.readouterr().err
