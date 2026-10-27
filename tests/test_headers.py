import json

from plexus.config import ExperimentConfig
from plexus.ingest.headers import chunk_header, clean_company, index_text
from plexus.retrieve.pipeline import BM25Retriever


def test_clean_company_drops_state_suffix_and_shouting():
    assert clean_company("QUALCOMM INC/DE") == "Qualcomm Inc"
    assert clean_company("DIODES INC /DEL/") == "Diodes Inc"
    assert clean_company("LAM RESEARCH CORP") == "Lam Research Corp"
    assert clean_company("KLA CORP") == "KLA Corp"
    assert clean_company("Broadcom Inc.") == "Broadcom Inc."  # mixed case left alone


def test_header_names_company_alias_form_period_and_section():
    meta = {"ticker": "ON", "company": "ON SEMICONDUCTOR CORP", "form": "10-Q",
            "report_date": "2025-04-04"}  # fmt: skip
    h = chunk_header(meta, "risk_factors")
    assert (
        h
        == "ON Semiconductor Corp (ON, onsemi) 10-Q for the period ended 2025-04-04. Section: Risk Factors."
    )
    assert index_text(h, "We rely on TSMC.").endswith("\nWe rely on TSMC.")
    assert index_text("", "x") == "x"


def test_bm25_matches_company_name_via_header(tmp_path):
    chunks = [
        {"chunk_id": "A#s0", "doc_id": "A", "start": 0, "end": 1, "section": "risk_factors",
         "text": "We depend on a single foundry.", "header": "Lam Research Corp (LRCX) 10-K."},
        {"chunk_id": "B#s0", "doc_id": "B", "start": 0, "end": 1, "section": "risk_factors",
         "text": "We depend on a single foundry.", "header": "KLA Corp (KLAC) 10-K."},
    ]  # fmt: skip
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    hits = BM25Retriever.from_index_dir(tmp_path).retrieve("Lam Research single foundry", 2)
    assert hits[0].chunk_id == "A#s0" and hits[0].header.startswith("Lam Research")


def test_headers_change_the_index_key():
    plain = ExperimentConfig.model_validate({"name": "a", "chunker": {"type": "structured"}})
    hdr = ExperimentConfig.model_validate(
        {"name": "b", "chunker": {"type": "structured", "headers": True}}
    )
    assert plain.index_key != hdr.index_key and hdr.index_key.endswith(
        "-hdr1__BAAI_bge-small-en-v1.5"
    )
