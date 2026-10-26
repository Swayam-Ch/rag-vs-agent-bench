"""Fixtures shared by several test files."""

import pytest

from ragbench.pipeline.chunk import main as chunk_main
from ragbench.pipeline.ingest import ingest_dir
from ragbench.pipeline.metadata import PaperMeta


@pytest.fixture
def processed(tmp_path):
    """data/processed built by the real ingest and chunk commands."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1706.03762.txt").write_text(
        "Attention Is All You Need\n\nThe Transformer relies entirely on attention.\n\n"
        "In Table 3 rows (A), we vary the number of attention heads and measure BLEU."
    )
    (raw / "2004.04906.txt").write_text(
        "Dense Passage Retrieval\n\nWe encode questions and passages with BERT.\n\n"
        "Retrieval accuracy improves over BM25 on Natural Questions."
    )
    meta = {
        "1706.03762": PaperMeta(
            "1706.03762", "v7", "Attention Is All You Need", ["A"], "2017-06-12", "cs.CL"
        ),
        "2004.04906": PaperMeta(
            "2004.04906", "v3", "Dense Passage Retrieval", ["B"], "2020-04-10", "cs.CL"
        ),
    }
    out = tmp_path / "processed"
    ingest_dir(raw, out, metadata=meta)
    assert chunk_main([str(out)]) == 0  # the default: paragraph, 1000
    return out
