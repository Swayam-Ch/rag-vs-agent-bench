import json
from pathlib import Path

import pytest

from ragbench.pipeline.ingest import (
    guess_title,
    ingest_dir,
    ingest_file,
    main,
    normalize_text,
    strip_references,
)

# --------------------------------------------------------------------------- #
# normalize_text: one test per cleaning rule, so a failure tells you which rule broke
# --------------------------------------------------------------------------- #


def test_ligatures_are_expanded():
    assert normalize_text("eﬃcient ﬁne-tuning") == "efficient fine-tuning"


def test_hyphenated_line_breaks_are_rejoined():
    assert normalize_text("dense retrie-\nval works") == "dense retrieval works"


def test_real_hyphens_are_kept():
    # "fine-tuning" has no line break, so the hyphen must survive
    assert normalize_text("fine-tuning") == "fine-tuning"


def test_soft_wraps_become_spaces_but_paragraphs_survive():
    raw = "first line\nsame paragraph\n\n\n\nsecond paragraph"
    assert normalize_text(raw) == "first line same paragraph\n\nsecond paragraph"


def test_whitespace_is_collapsed_and_stripped():
    assert normalize_text("  a   \t b  \r\n") == "a b"


def test_guess_title_falls_back_to_stem():
    assert guess_title("# Attention Is All You Need\n\nbody", "x") == "Attention Is All You Need"
    assert guess_title("", "paper_01") == "paper_01"


# --------------------------------------------------------------------------- #
# ingest_file / ingest_dir
# --------------------------------------------------------------------------- #


def _make_pdf(path: Path, pages: list[str]) -> None:
    """Write a tiny multi-page PDF so tests don't depend on downloaded files."""
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


def test_ingest_txt(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("My Title\nwrapped\nline\n\nNext para", encoding="utf-8")

    doc = ingest_file(src)

    assert doc.title == "My Title wrapped line"  # soft wraps joined -> one line
    assert doc.text.endswith("Next para")
    assert doc.page_offsets == [0]
    assert len(doc.doc_id) == 12


def test_ingest_pdf_tracks_page_offsets(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf, ["Page one about transformers", "Page two about retrieval"])

    doc = ingest_file(pdf)

    assert len(doc.page_offsets) == 2
    second_page = doc.text[doc.page_offsets[1] :]
    assert second_page.startswith("Page two")


def test_doc_id_is_deterministic(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same content")
    b.write_text("same content")
    assert ingest_file(a).doc_id == ingest_file(b).doc_id


def test_ingest_dir_writes_markdown_and_manifest(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    (raw / "one.txt").write_text("Doc one")
    (raw / "two.md").write_text("# Doc two")
    (raw / "ignored.png").write_bytes(b"\x89PNG")

    docs = ingest_dir(raw, out)

    assert len(docs) == 2
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert {m["source"] for m in manifest} == {"one.txt", "two.md"}
    assert all("text" not in m for m in manifest)  # body stays out of the manifest
    for m in manifest:
        body = (out / f"{m['doc_id']}.md").read_text()
        assert not body.startswith("---")  # no YAML front matter in searchable text


def test_cli_errors_cleanly_on_empty_dir(tmp_path, capsys):
    assert main([str(tmp_path), str(tmp_path / "out")]) == 1
    assert "No" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# strip_references
# --------------------------------------------------------------------------- #

BODY = "5 Conclusion\nWe show that retrieval helps.\n"
REFS = "References\n[1] A. Author. A Paper. 2020.\n[2] B. Author. Another. 2021.\n"


def test_strip_references_basic():
    pages, removed = strip_references([BODY + REFS])
    assert pages == [BODY]
    assert removed == len(REFS)


def test_strip_references_without_heading_is_a_no_op():
    pages, removed = strip_references([BODY])
    assert (pages, removed) == ([BODY], 0)


def test_word_references_mid_sentence_is_not_a_heading():
    text = "This section lists references to prior work.\nReferences are below.\n"
    assert strip_references([text]) == ([text], 0)


@pytest.mark.parametrize("heading", ["REFERENCES", "7 References", "8. References", "Bibliography"])
def test_heading_variants(heading):
    pages, removed = strip_references([BODY + heading + "\n[1] X.\n"])
    assert pages == [BODY]
    assert removed > 0


def test_cuts_from_the_last_heading():
    # an early line that happens to be just "References" (e.g. in a table of contents)
    text = "Contents\nReferences\nIntro text.\n" + BODY + REFS
    pages, _ = strip_references([text])
    assert pages == ["Contents\nReferences\nIntro text.\n" + BODY]


def test_references_spanning_pages_keep_page_count():
    pages, _ = strip_references(["Intro.\n", BODY + REFS, "[3] C. Author. 2022.\n"])
    assert pages == ["Intro.\n", BODY, ""]  # 3 pages in, 3 pages out


def test_appendix_after_references_is_kept():
    appendix = "A Appendix: Hyperparameters\nWe use a learning rate of 1e-4.\n"
    pages, _ = strip_references([BODY + REFS, "[3] More refs.\nAppendix\n" + appendix])
    assert pages == [BODY, "Appendix\n" + appendix]


def test_ingest_strips_references_by_default(tmp_path):
    src = tmp_path / "paper.txt"
    src.write_text(BODY + REFS)
    doc = ingest_file(src)
    assert "A Paper" not in doc.text
    assert doc.references_removed > 0
    assert doc.manifest_entry()["references_removed"] == doc.references_removed

    kept = ingest_file(src, keep_references=True)
    assert "A Paper" in kept.text
    assert kept.references_removed == 0
