from plexus import doctor


def test_anthropic_key_check_fails_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert not doctor.check_anthropic_key_unset().ok
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert doctor.check_anthropic_key_unset().ok


def test_sec_user_agent_needs_name_and_email(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "justaname")
    assert not doctor.check_sec_user_agent().ok
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert doctor.check_sec_user_agent().ok


def test_later_phase_failures_do_not_fail_phase_0(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "CHECKS",
        [
            doctor.Check("ok", 0, lambda: doctor.Result(True, "fine")),
            doctor.Check("later", 3, lambda: doctor.Result(False, "nope", "do X")),
        ],
    )
    assert doctor.main(["--phase", "0"]) == 0
    assert doctor.main(["--phase", "3"]) == 1
    out = capsys.readouterr().out
    assert "do X" in out and "[TODO]" in out and "[FAIL]" in out


def test_version_parsing():
    assert doctor._version("v22.15.0") == (22, 15, 0)
    assert doctor._version("git version 2.53.0") == (2, 53, 0)
    assert doctor._version("garbage") == (0,)
