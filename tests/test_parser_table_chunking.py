"""Chunking behaviour for sparse Office tables and oversized lines."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.parser import DocumentParser


def make_parser() -> DocumentParser:
    # __new__ skips __init__: these tests only exercise pure splitting logic
    # and must not pay Docling/EasyOCR model initialization.
    return DocumentParser.__new__(DocumentParser)


def sparse_row(index: int, columns: int = 120, filled: int = 4) -> str:
    cells = [""] * columns
    step = columns // filled
    for position in range(filled):
        cells[position * step] = f"値{index}-{position}"
    return "| " + " | ".join(cells) + " |"


def test_sparse_excel_table_rows_are_compacted():
    parser = make_parser()
    header = "| 番号 | " + " | ".join([""] * 118) + " | 決裁者 |"
    separator = "|" + "---|" * 120
    rows = [sparse_row(i) for i in range(40)]
    body = "\n".join(["決裁権限基準表", header, separator, *rows])

    chunks = parser._split_markdown_table(
        body,
        filename="kessai.xlsx",
        page_number=1,
        start_index=0,
        metadata={},
    )

    assert chunks
    for chunk in chunks:
        # Compaction target: no chunk anywhere near the raw sparse size
        # (a single raw row is ~700 chars of pipes for 4 values).
        assert len(chunk.text) <= parser.TABLE_CHUNK_CHAR_BUDGET + 400
    # Every value survives compaction.
    joined = "\n".join(chunk.text for chunk in chunks)
    for i in range(40):
        assert f"値{i}-0" in joined


def wide_row(values: dict[int, str], columns: int = 120) -> str:
    """Build a sparse row like Excel's merged-cell export: mostly empty
    cells (enough to push length past the 300-char compaction threshold)
    with a few real values at given column positions."""
    cells = [""] * columns
    for position, value in values.items():
        cells[position] = value
    return "| " + " | ".join(cells) + " |"


def test_compacted_row_keeps_column_name_next_to_its_value():
    """A bare compacted value ("1億円以上") loses which column it came from —
    the reader can no longer tell a threshold from an approver. Labelling
    ("基準: 1億円以上") preserves that meaning through compaction."""
    parser = make_parser()
    header = wide_row({0: "項目Ｎｏ", 41: "基準", 83: "決裁者"})
    separator = "|" + "---|" * 120
    raw_row = wide_row(
        {0: "Ⅴ-3②", 41: "1億円以上、5億円未満のもの", 83: "本社取締役会"}
    )
    body = "\n".join([header, separator, raw_row])

    compacted = parser._split_markdown_table(
        body, filename="kessai.xlsx", page_number=1, start_index=0, metadata={},
    )[0].text

    assert "基準: 1億円以上、5億円未満のもの" in compacted
    assert "決裁者: 本社取締役会" in compacted
    assert "項目Ｎｏ: Ⅴ-3②" in compacted
    # Bare, unlabelled values (the old behaviour) must not reappear alone.
    assert "| 1億円以上、5億円未満のもの |" not in compacted


def test_compact_table_row_without_header_falls_back_to_bare_values():
    parser = make_parser()
    row = wide_row({1: "値A", 100: "値B"})

    assert parser._compact_table_row(row) == "| 値A | 値B |"
    header_cells = [""] * 120
    header_cells[1], header_cells[100] = "列A", "列B"
    assert (
        parser._compact_table_row(row, header_cells=header_cells)
        == "| 列A: 値A | 列B: 値B |"
    )


def test_dense_table_rows_keep_original_alignment():
    parser = make_parser()
    header = "| ID | Name | Dept |"
    separator = "|---|---|---|"
    rows = [f"| {i} | Person {i} | Dept {i % 3} |" for i in range(30)]
    body = "\n".join([header, separator, *rows])

    chunks = parser._split_markdown_table(
        body,
        filename="roster.md",
        page_number=1,
        start_index=0,
        metadata={},
    )

    assert chunks
    # 30 data rows at 15 rows/chunk -> exactly 2 chunks, rows untouched.
    assert len(chunks) == 2
    assert "| 3 | Person 3 | Dept 0 |" in chunks[0].text


def test_table_chunks_never_exceed_char_budget_even_at_15_rows():
    parser = make_parser()
    wide = "x" * 700
    header = "| A | B |"
    separator = "|---|---|"
    rows = [f"| {wide} | {i} |" for i in range(20)]
    body = "\n".join([header, separator, *rows])

    chunks = parser._split_markdown_table(
        body,
        filename="wide.xlsx",
        page_number=1,
        start_index=0,
        metadata={},
    )

    assert len(chunks) > 2  # char budget forces more, smaller chunks
    for chunk in chunks:
        assert len(chunk.text) <= parser.TABLE_CHUNK_CHAR_BUDGET + 800


def test_split_text_hard_splits_single_oversized_line():
    parser = make_parser()
    one_line = "あ" * (DocumentParser.CHUNK_SIZE * 3 + 100)

    chunks = parser._split_text(
        one_line,
        filename="minified.txt",
        page_number=1,
        start_index=0,
        metadata={},
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= DocumentParser.CHUNK_SIZE + DocumentParser.CHUNK_OVERLAP + 1
    assert sum(chunk.text.count("あ") for chunk in chunks) >= len(one_line)


def test_mid_table_stray_lines_do_not_bloat_first_chunk():
    parser = make_parser()
    header = "| 項目 | 決裁者 |"
    separator = "|---|---|"
    rows = [f"| 項目{i} | 部門長{i} |" for i in range(30)]
    # A sparse Excel export inserts huge pipe-bearing lines *between* rows
    # (cell continuations that Docling failed to attach to any row).
    stray = "28版：2025年4月21日改定 | " + " | ".join([""] * 1200) + " | Ⅸ-3⑥"
    lines = [header, separator, *rows[:10], stray, *rows[10:]]
    body = "タイトル\n" + "\n".join(lines)

    chunks = parser._split_markdown_table(
        body,
        filename="kessai.xlsx",
        page_number=1,
        start_index=0,
        metadata={},
    )

    assert chunks
    for chunk in chunks:
        assert len(chunk.text) <= parser.TABLE_CHUNK_CHAR_BUDGET + 800
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "28版：2025年4月21日改定" in joined  # nội dung thật vẫn giữ
    assert "項目29" in joined


def test_docling_markdown_with_big_table_routes_to_table_splitter():
    parser = make_parser()
    header = "| Col |" + " |" * 119
    separator = "|" + "---|" * 120
    rows = [sparse_row(i) for i in range(30)]
    body = "\n".join([header, separator, *rows])

    chunks = parser._split_page_body(
        body,
        filename="sheet.xlsx",
        page_number=1,
        start_index=0,
        metadata={},
    )

    assert chunks
    assert all(len(chunk.text) <= parser.TABLE_CHUNK_CHAR_BUDGET + 400 for chunk in chunks)
    assert all(chunk.content_type == "table" for chunk in chunks)
