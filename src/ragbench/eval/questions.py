"""The benchmark's question set: format, loading and validation.

`eval/questions.yaml` is the answer key the methods are scored against, so it must
never be silently wrong. Loading checks every question and reports *all* problems at
once, each with its line number in the file.

A question looks like this:

    - id: q001
      type: single_fact
      question: How many attention heads does the base Transformer use?
      answer: 8 parallel attention heads.
      gold:
        - paper: "1706.03762"
          pages: [5]
      notes: optional; why the question is interesting or hard
      author: optional; who wrote it (e.g. swayam, claude), to split results by author

`gold` pairs each paper with the pages that contain its evidence. The pages are
alternatives: finding any one of them counts as finding that evidence (see
metrics.py). A question needing two separate facts from one paper lists it twice.
Question types:

* single_fact   the answer is in one place in one paper
* multi_filter  "which papers do X (and Y)": the answer is the set of papers in `gold`
* cross_doc     combining facts from two or more papers (compare, contrast)
* negation      "which papers do NOT ...": the set of papers that don't. Absence has no
                page to point at, so these entries may say `pages: []`, meaning "the
                whole paper was checked"; they are scored on papers only.
* unanswerable  the papers don't contain the answer; `gold` is empty, and the right
                response is to say so

Paper IDs are arXiv IDs from data/sources.txt. Quote them in YAML ("2005.11401"),
otherwise YAML reads 2005.11401 as a number and 1810.04805 loses its trailing zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FORMAT_VERSION = 1
TYPES = ("single_fact", "multi_filter", "cross_doc", "negation", "unanswerable")
QUESTION_FIELDS = {"id", "type", "question", "answer", "gold", "notes", "author"}
GOLD_FIELDS = {"paper", "pages"}
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")


@dataclass(frozen=True)
class Evidence:
    paper: str  # arXiv ID, e.g. "1706.03762"
    pages: tuple[int, ...]


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    answer: str
    gold: tuple[Evidence, ...] = ()
    notes: str = ""
    author: str = ""
    line: int = 0  # where it starts in the YAML file, for error messages

    @property
    def gold_papers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.paper for e in self.gold))


@dataclass
class QuestionSet:
    questions: list[Question] = field(default_factory=list)

    def by_type(self) -> dict[str, int]:
        counts = dict.fromkeys(TYPES, 0)
        for q in self.questions:
            counts[q.type] += 1
        return counts


class QuestionSetError(ValueError):
    """Raised with every problem found, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def _item_lines(text: str) -> list[int]:
    """1-based line where each entry of `questions:` starts, read from the YAML node tree."""
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return []
    if not isinstance(root, yaml.MappingNode):
        return []
    for key, value in root.value:
        if key.value == "questions" and isinstance(value, yaml.SequenceNode):
            return [item.start_mark.line + 1 for item in value.value]
    return []


def parse(text: str, source: str = "questions.yaml") -> QuestionSet:
    """Parse and validate the YAML text. Raises QuestionSetError listing every problem."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise QuestionSetError([f"{source}: not valid YAML: {exc}"]) from exc
    if not isinstance(data, dict):
        raise QuestionSetError([f"{source}: expected a mapping with 'version' and 'questions'"])
    problems: list[str] = []
    if data.get("version") != FORMAT_VERSION:
        problems.append(
            f"{source}: 'version' must be {FORMAT_VERSION}, got {data.get('version')!r}"
        )
    items = data.get("questions")
    if not isinstance(items, list) or not items:
        raise QuestionSetError(problems + [f"{source}: 'questions' must be a non-empty list"])
    extra = set(data) - {"version", "questions"}
    if extra:
        problems.append(f"{source}: unknown top-level keys: {', '.join(sorted(extra))}")

    lines = _item_lines(text)
    questions: list[Question] = []
    seen_ids: dict[str, int] = {}
    for index, item in enumerate(items):
        line = lines[index] if index < len(lines) else 0
        where = f"{source}:{line}" if line else f"{source} (question {index + 1})"
        question, item_problems = _parse_question(item, where, line)
        if question is not None:
            if question.id in seen_ids:
                item_problems.append(
                    f"{where}: duplicate id {question.id!r} (first used on line "
                    f"{seen_ids[question.id]})"
                )
            seen_ids.setdefault(question.id, line)
            questions.append(question)
        problems += item_problems

    if problems:
        raise QuestionSetError(problems)
    return QuestionSet(questions)


def _parse_question(item: object, where: str, line: int) -> tuple[Question | None, list[str]]:
    if not isinstance(item, dict):
        return None, [f"{where}: each question must be a mapping (id, type, question, ...)"]
    qid = str(item.get("id", "")).strip()
    where = f"{where}: {qid}" if qid else where
    problems: list[str] = []

    unknown = set(item) - QUESTION_FIELDS
    if unknown:
        problems.append(f"{where}: unknown fields: {', '.join(sorted(map(str, unknown)))}")
    if not _ID.match(qid):
        problems.append(f"{where}: 'id' must be lowercase letters, digits, - or _ (e.g. q001)")
    qtype = item.get("type")
    if qtype not in TYPES:
        problems.append(f"{where}: unknown type {qtype!r}; expected one of: {', '.join(TYPES)}")
    for key in ("question", "answer"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            problems.append(f"{where}: '{key}' must be a non-empty text")

    gold, gold_problems = _parse_gold(item.get("gold", []), where, qtype)
    problems += gold_problems
    papers = {e.paper for e in gold}
    if qtype == "unanswerable" and gold:
        problems.append(f"{where}: an unanswerable question must have an empty 'gold'")
    elif qtype in TYPES and qtype != "unanswerable" and not gold:
        problems.append(f"{where}: 'gold' must list at least one paper with its pages")
    if qtype == "cross_doc" and len(papers) < 2:
        problems.append(f"{where}: a cross_doc question needs evidence from at least 2 papers")

    notes = item.get("notes", "")
    if not isinstance(notes, str):
        problems.append(f"{where}: 'notes' must be text")
    author = item.get("author", "")
    if not isinstance(author, str):
        problems.append(f"{where}: 'author' must be text, e.g. swayam")
    if problems:
        return None, problems
    return (
        Question(
            id=qid,
            type=qtype,
            question=item["question"].strip(),
            answer=item["answer"].strip(),
            gold=tuple(gold),
            notes=notes.strip(),
            author=author.strip(),
            line=line,
        ),
        [],
    )


def _parse_gold(raw: object, where: str, qtype: object = None) -> tuple[list[Evidence], list[str]]:
    if not isinstance(raw, list):
        return [], [f"{where}: 'gold' must be a list of {{paper, pages}} entries"]
    evidence, problems = [], []
    for n, entry in enumerate(raw, start=1):
        label = f"{where}: gold entry {n}"
        if not isinstance(entry, dict):
            problems.append(f"{label}: must be a mapping with 'paper' and 'pages'")
            continue
        unknown = set(entry) - GOLD_FIELDS
        if unknown:
            problems.append(f"{label}: unknown fields: {', '.join(sorted(map(str, unknown)))}")
        paper = entry.get("paper")
        if isinstance(paper, float):
            problems.append(
                f"{label}: paper {paper!r} was read as a number; put the arXiv ID in quotes"
            )
            continue
        paper = str(paper or "").strip()
        if not _ARXIV.match(paper):
            problems.append(f'{label}: paper {paper!r} is not an arXiv ID like "1706.03762"')
        pages = entry.get("pages")
        if qtype == "negation" and pages == []:
            # "this paper never does X": checked across the whole paper, no single page
            evidence.append(Evidence(paper, ()))
            continue
        if not isinstance(pages, list) or not pages:
            problems.append(
                f"{label}: 'pages' must be a non-empty list of page numbers"
                " (only a negation question may use [] for 'the whole paper')"
            )
            continue
        if not all(isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in pages):
            problems.append(f"{label}: pages must be whole numbers from 1 up, got {pages!r}")
            continue
        evidence.append(Evidence(paper, tuple(sorted(set(pages)))))
    return evidence, problems


def check_against_corpus(
    questions: QuestionSet, known_papers: set[str], page_counts: dict[str, int] | None = None
) -> list[str]:
    """Problems with papers not in the corpus or pages past a paper's end.

    `known_papers`: arXiv IDs of the corpus (from data/metadata.jsonl, committed).
    `page_counts`: arXiv ID -> number of pages (from data/processed/manifest.jsonl, only
    available after ingestion), to catch page numbers that don't exist.
    """
    problems = []
    for q in questions.questions:
        for e in q.gold:
            where = f"line {q.line}: {q.id}"
            if e.paper not in known_papers:
                problems.append(f"{where}: paper {e.paper} is not in the corpus")
            elif (
                page_counts
                and e.pages
                and e.paper in page_counts
                and max(e.pages) > page_counts[e.paper]
            ):
                problems.append(
                    f"{where}: page {max(e.pages)} of {e.paper} doesn't exist "
                    f"(the paper has {page_counts[e.paper]} pages)"
                )
    return problems


def load(path: Path) -> QuestionSet:
    return parse(path.read_text(encoding="utf-8"), source=path.name)
