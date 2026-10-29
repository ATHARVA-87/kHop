from collections import Counter

from plexus.serve.reframe import Reframer

VOCAB = Counter({"what": 500, "does": 500, "say": 300, "about": 900, "revenue": 400, "growth": 300,
                 "nvidia": 200, "capex": 50, "onto": 40, "how": 500, "performed": 30, "in": 9000,
                 "the": 9000, "are": 800, "companies": 600, "company": 900, "describe": 100,
                 "distributors": 120, "dependence": 60})  # fmt: skip
FORMS = {"NVIDIA": "NVIDIA", "Micron Technology": "Micron Technology", "Micron": "Micron Technology",
         "Lattice Semiconductor": "Lattice Semiconductor", "Texas Instruments": "Texas Instruments",
         "TI": "Texas Instruments", "Skyworks Solutions": "Skyworks Solutions", "Amkor Technology": "Amkor Technology"}  # fmt: skip
R = Reframer(VOCAB, FORMS)


def test_three_owner_phrasings_become_one_question():
    a = R("How are the nvidia company performed in 2024").question
    b = R("How are the nvidia companies performed in 2024").question
    c = R("How are the nvdia companies performed in 2024").question
    assert a == b == c == "How are NVIDIA performed in 2024"


def test_spelling_fixes_names_and_words():
    out = R("What does Micorn say about revenue growht?")
    assert out.question == "What does Micron say about revenue growth?"
    assert {c["kind"] for c in out.changes} == {"spelling"} and out.changed


def test_never_touches_traps_short_tokens_numbers_or_quotes():
    for q in [
        "What does TI say about capex?",
        "What does ON say about onto in FY2024?",
        'What does "Nvdia" mean in 10-K filings?',
        "How does Amkor describe HBM3E?",
    ]:
        r = R(q)
        assert not r.changed, (q, r.question)


def test_case_only_change_is_not_shown_as_a_change():
    r = R("what does nvidia say about revenue?")
    assert r.question == "what does NVIDIA say about revenue?" and not r.changed


def test_v2_fixes_possessives_mixed_case_names_and_valid_words():
    r2 = Reframer(
        VOCAB | Counter({"does": 5000, "dies": 900, "diodes": 300}), FORMS | {"Diodes": "Diodes"}
    )
    assert not r2("What does Texas Instruments' filing say?").changed
    assert (
        r2("What does NVIDbA say about revenue?").question == "What does NVIDIA say about revenue?"
    )
    assert (
        r2("What does Dioes say about revenue?").question == "What does Diodes say about revenue?"
    )
    assert (
        not r2("Why is revenue falling at NVIDIA?").changed
        or "falling" in r2("Why is revenue falling at NVIDIA?").question
    )


def test_trailing_inc_only_as_a_whole_word():
    r = Reframer(VOCAB | Counter({"expenses": 300, "include": 900, "research": 800, "development": 700}),
                 FORMS | {"Silicon Labs": "Silicon Labs"})  # fmt: skip
    q = "What expenses does Silicon Labs include in research and development?"
    assert not r(q).changed and r(q).question == q
    assert (
        r("What does Silicon Labs Inc. say about revenue?").question
        == "What does Silicon Labs say about revenue?"
    )


def test_v3_short_targets_question_words_and_three_letter_typos():
    v = VOCAB | Counter(
        {
            "for": 9000,
            "four": 600,
            "that": 20000,
            "from": 8000,
            "have": 5000,
            "type": 300,
            "know": 900,
            "show": 800,
        }
    )
    r3 = Reframer(v, FORMS)
    assert r3("Ihat does NVIDIA say about revenue?").question.startswith("What does")
    assert "for" in r3("What does NVIDIA say fovr revenue?").question.split()
    assert (
        r3("What revenue fom NVIDIA?").question == "What revenue fom NVIDIA?"
    )  # from/for: ambiguous, left alone
    assert (
        r3("What does NVIDIA hve about revenue?").question == "What does NVIDIA have about revenue?"
    )
    assert not r3("What does TSM say about KLA?").changed  # 3-letter capitals are codes
