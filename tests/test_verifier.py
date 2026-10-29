from plexus.agents import verifier as v


class FakeScorer:
    def score(self, doc, claim):
        return 0.9 if claim.split()[0].lower() in doc.lower() else 0.1


def test_split_and_strip():
    s = v.split_sentences(
        "TSMC makes chips [1]. Revenue was $1,200 million [2].\n- **Risk:** supply [1]."
    )
    assert len(s) == 3 and v.strip_citations(s[2]) == "Risk: supply."
    assert v.numbers("Revenue was $1,200.5 million in 2025.") == {"1200.5", "2025"}


def test_check_answer_rules():
    passages = ["TSMC makes most of the chips in Taiwan.", "Revenue was 1,200 million dollars."]
    ans = "TSMC makes most of our chips [1]. Revenue was 9,999 million dollars [2]. Margins rose sharply this year overall. Intel builds factories everywhere [7]."
    out = {c.sentence.split()[0]: c for c in v.check_answer(ans, passages, FakeScorer())}
    assert out["TSMC"].supported
    assert out["Revenue"].issues == [] and out["Revenue"].supported  # MiniCheck decides
    assert out["Margins"].issues == ["no_citation"]
    assert out["Intel"].issues == ["bad_citation"]


def test_splitter_review_cases():
    s = v.split_sentences(
        "The U.S. Department of Commerce added rules [1]. Revenue rose. [2] Next item [3]."
    )
    assert s[0].startswith("The U.S. Department") and s[0].endswith("[1].")
    assert s[1] == "Revenue rose. [2]" and s[2] == "Next item [3]."
