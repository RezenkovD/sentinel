"""Thesis pack (session 06, step 4): the numbers file and its formatting, without models or matplotlib."""

from pathlib import Path

from sentinel.eval.thesis_pack import Numbers, _ci, _pct, write_table


def test_every_number_names_its_source_and_the_files_date(tmp_path: Path):
    source = tmp_path / "result.json"
    source.write_text("{}")
    numbers = Numbers()
    numbers.section("Розділ")
    numbers.add("AUC", _ci(0.8812, [0.845, 0.9101]), source)
    numbers.add("Тестів", "383", "pytest --collect-only")
    text = numbers.text("Ключові числа")
    assert "- AUC: **0,881 [0,845; 0,910]** — `" in text and str(source) in text
    assert text.count(".2026") == 1  # a file source carries its date, a command does not
    assert _pct(0.005) == "0,5 %"


def test_tables_come_as_markdown_and_csv_with_the_same_ukrainian_header(tmp_path: Path):
    write_table(tmp_path, "t", "Таблиця. Тест", ["Метод", "AUC"], [["A", "0,881"]])
    assert (tmp_path / "t.md").read_text(encoding="utf-8").startswith("**Таблиця. Тест**")
    assert (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines() == ["Метод,AUC", 'A,"0,881"']
