from plexus import readme


def test_readme_tables_match_results():
    """Every number in README.md is generated from results/*.json (rule 9)."""
    assert readme.main(["--check"]) == 0


def test_unknown_marker_is_an_error():
    import pytest

    with pytest.raises(KeyError):
        readme.render("<!-- table:nope -->\n<!-- /table -->")


def test_drift_table_from_a_result(tmp_path, monkeypatch):
    import json

    stats = {"diff": {"mean": -0.05, "low": -0.1, "high": 0.0}}
    ci = {"mean": 0.6, "low": 0.5, "high": 0.7, "n": 40}
    (tmp_path / "drift.json").write_text(
        json.dumps(
            {
                "configs": {"old": "a3h_rerank", "new": "a3h_rerank_2026"},
                "corpus": {
                    "old": {"filings": 156, "companies": 39, "by_form": {"10-K": 39, "10-Q": 117}},
                    "new": {"filings": 140, "companies": 38, "by_form": {"10-K": 38, "10-Q": 102}},
                },
                "golden_questions": 225,
                "carry_over": {"all": {"carried": 40, "gone": 185}},
                "search_on_carried_questions": {
                    "recall@10": {"questions": 40, "old": ci, "new": ci, "new_minus_old": stats}
                },
            }
        )
    )
    monkeypatch.setattr(readme, "RESULTS", tmp_path)
    table = readme.drift()
    assert "Of 225 golden questions, 40 have every evidence quote" in table
    assert "| a3h_rerank_2026 | 140 | 38 10-K, 102 10-Q | 38 |" in table
    assert "-0.05 [-0.10, +0.00]" in table
