from plexus.eval.support import paired_diff, summarize


def test_summary_and_paired_diff():
    a = {
        "q1": {
            "refused": False,
            "n": 2,
            "supported": 1,
            "uncited": 0,
            "caret": False,
            "cited": 1,
            "cited_supported": 1,
        },
        "q2": {
            "refused": True,
            "n": 0,
            "supported": 0,
            "uncited": 0,
            "caret": False,
            "cited": 1,
            "cited_supported": 1,
        },
    }
    b = {
        "q1": {
            "refused": False,
            "n": 2,
            "supported": 2,
            "uncited": 0,
            "caret": True,
            "cited": 2,
            "cited_supported": 2,
        },
        "q2": {
            "refused": False,
            "n": 1,
            "supported": 1,
            "uncited": 0,
            "caret": False,
            "cited": 1,
            "cited_supported": 1,
        },
    }
    s = summarize(a)
    assert (
        s["sentence_support"] == 0.5 and s["fully_supported_answers"] == 0.0 and s["refused"] == 1
    )
    assert summarize(b)["caret_leak"] == 1
    d = paired_diff(a, b)
    assert d["n"] == 1 and d["diff_fully_supported"] == 1.0
