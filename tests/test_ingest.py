import json
from pathlib import Path

import pytest

from ragbench.pipeline.ingest import (
    _is_next_label,
    guess_title,
    ingest_dir,
    ingest_file,
    main,
    normalize_text,
    remove_repeated_lines,
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


# --- lettered appendices (seen in InstructGPT, ReAct, BEIR, Atlas) ---

LETTERED = "A Training details and additional results\nA.1 MMLU\nWe use 5-shot prompts.\n"


def test_lettered_appendix_is_kept():
    pages, _ = strip_references([BODY + REFS + LETTERED])
    assert pages == [BODY + LETTERED]


def test_appendix_starting_at_a1_is_kept():
    # ReAct's top-level "A" heading is mangled by PDF extraction; its outline starts at A.1
    appendix = "A.1 GPT-3 Experiments\nResults.\nA.2 Up-to-date knowledge\nMore.\n"
    pages, _ = strip_references([BODY + REFS, appendix])
    assert pages == [BODY, appendix]


def test_lettered_appendix_found_across_pages():
    pages, _ = strip_references([BODY + REFS, "[3] C. Author. 2022.\n" + LETTERED])
    assert pages == [BODY, LETTERED]


def test_author_initials_are_not_headings():
    refs = (
        "References\n"
        "A. Vaswani, N. Shazeer, N. Parmar. Attention. 2017.\n"
        "B. Smith. Retrieval. 2019.\n"
    )
    pages, _ = strip_references([BODY + refs])
    assert pages == [BODY]


def test_reference_title_line_is_not_an_appendix():
    # a wrapped reference that starts like a heading, with no outline after it
    refs = (
        "References\n[1] Lewis et al. 2020.\n"
        "A Survey of Dense Retrieval Methods\n"
        "[2] Izacard. 2021.\n"
    )
    pages, _ = strip_references([BODY + refs])
    assert pages == [BODY]


def test_reference_title_line_before_real_appendix():
    refs = "References\n[1] Lewis.\nA Survey of Dense Retrieval Methods\n[2] Izacard.\n"
    pages, _ = strip_references([BODY + refs + LETTERED])
    assert pages == [BODY + LETTERED]


@pytest.mark.parametrize(
    ("prev", "nxt", "ok"),
    [
        ("A", "A.1", True),
        ("A", "B", True),
        ("A.1", "A.2", True),
        ("A.1", "A.1.1", True),
        ("A.1", "B", True),
        ("A.2.2", "A.3", True),
        ("A", "A", False),
        ("A", "C", False),
        ("A.1", "A.3", False),
    ],
)
def test_is_next_label(prev, nxt, ok):
    assert _is_next_label(prev, nxt) is ok


# --------------------------------------------------------------------------- #
# remove_repeated_lines (running headers / footers / page numbers)
# --------------------------------------------------------------------------- #


def _paper(n_pages: int, header: str = "Published as a conference paper at ICLR 2023") -> list[str]:
    return [
        f"{header}\nBody text of page {i}.\nMore findings on page {i}.\n{i}"
        for i in range(1, n_pages + 1)
    ]


def test_running_header_and_page_numbers_removed():
    pages, removed = remove_repeated_lines(_paper(4))
    assert pages[0] == "Body text of page 1.\nMore findings on page 1."
    assert removed.count("Published as a conference paper at ICLR 2023") == 4
    assert len(removed) == 8  # 4 headers + 4 page numbers


@pytest.mark.parametrize("header", ["Under review as a conference paper {i}", "{i} Preprint"])
def test_header_with_page_number_counts_as_repeated(header):
    pages = [header.format(i=i) + f"\nContent {i}." for i in range(1, 5)]
    out, _ = remove_repeated_lines(pages)
    assert out == [f"Content {i}." for i in range(1, 5)]


def test_numbers_inside_body_lines_still_count():
    pages = [f"Results on dataset {i} are strong.\nOther text {i}." for i in range(4)]
    assert remove_repeated_lines(pages) == (pages, [])


def test_header_on_too_few_pages_is_kept():
    pages = ["Special note\nA."] + [f"Body {i}." for i in range(5)]
    out, removed = remove_repeated_lines(pages)
    assert out[0] == "Special note\nA."
    assert removed == []


def test_repeated_line_in_middle_of_page_is_kept():
    middle = "As shown in Table 1, retrieval helps."
    words = ["alpha", "beta", "gamma", "delta"]
    pages = [
        "\n".join(
            [f"{w} one", f"{w} two", f"{w} three", middle, f"{w} five", f"{w} six", f"{w} seven"]
        )
        for w in words
    ]
    out, removed = remove_repeated_lines(pages)
    assert all(middle in page for page in out)
    assert removed == []


@pytest.mark.parametrize("template", ["{n}", "Page {n}", "{n} of 12", "{n}/12"])
def test_page_number_formats(template):
    pages = [f"Text {n}.\n" + template.format(n=n) for n in (1, 2, 3)]
    out, removed = remove_repeated_lines(pages)
    assert out == ["Text 1.", "Text 2.", "Text 3."]
    assert len(removed) == 3


def test_unnumbered_cover_page_shifts_the_sequence():
    # cover has no number; page 2 of the PDF prints "1", page 3 prints "2", ...
    pages = ["Title page."] + [f"Body {n}.\n{n}" for n in (1, 2, 3, 4)]
    out, removed = remove_repeated_lines(pages)
    assert out == ["Title page.", "Body 1.", "Body 2.", "Body 3.", "Body 4."]
    assert removed == ["1", "2", "3", "4"]


def test_chart_tick_labels_at_page_edge_are_kept():
    # seen in Chain-of-Thought / Lost in the Middle: a figure's axis at the bottom of a page
    pages = [f"Body {n}.\n{n}" for n in (1, 2, 3, 4)]
    pages[1] = "Figure 2: accuracy.\n0\n20\n40\n2"
    out, removed = remove_repeated_lines(pages)
    assert out[1] == "Figure 2: accuracy.\n0\n20\n40"  # ticks kept, page number "2" gone
    assert removed == ["1", "2", "3", "4"]


def test_numbers_that_never_follow_the_pages_are_kept():
    rows = [("Recall on NQ", "44"), ("Recall on TQA", "65"), ("EM", "50"), ("F1", "50")]
    pages = [f"{caption}\n{value}" for caption, value in rows]
    assert remove_repeated_lines(pages) == (pages, [])


def test_lone_number_in_single_page_file_is_kept():
    assert remove_repeated_lines(["2020\nA year to remember."]) == (
        ["2020\nA year to remember."],
        [],
    )


def test_ingest_records_boilerplate_count(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf, ["Header Line", "Header Line", "Header Line"])
    doc = ingest_file(pdf)
    assert doc.boilerplate_lines_removed == 3
    assert "Header Line" not in doc.text
