import os
import sys

from polycast.cli import main

TEXT = "Will the central bank cut the policy rate at the next scheduled meeting?"
CRITERIA = "Resolved yes if the published decision confirms a reduction."


def run(monkeypatch, argv) -> int:
    monkeypatch.setattr(sys, "argv", ["polycast", *argv])
    return main()


def test_shared_options_work_before_the_subcommand(tmp_path, monkeypatch, capsys):
    code = run(
        monkeypatch,
        [
            "--db", str(tmp_path / "a.db"),
            "--audit-dir", str(tmp_path / "runs"),
            "forecast",
            "--text", TEXT,
            "--criteria", CRITERIA,
            "--resolve-by", "2026-12-18",
        ],
    )
    assert code == 0
    assert "probability" in capsys.readouterr().out
    assert (tmp_path / "a.db").exists()


def test_shared_options_work_after_the_subcommand(tmp_path, monkeypatch, capsys):
    code = run(
        monkeypatch,
        [
            "forecast",
            "--text", TEXT,
            "--criteria", CRITERIA,
            "--resolve-by", "2026-12-18",
            "--db", str(tmp_path / "b.db"),
            "--audit-dir", str(tmp_path / "runs2"),
        ],
    )
    assert code == 0
    assert (tmp_path / "b.db").exists()


def test_subcommand_value_wins_over_the_top_level_one(tmp_path, monkeypatch, capsys):
    run(
        monkeypatch,
        [
            "--db", str(tmp_path / "top.db"),
            "forecast",
            "--text", TEXT,
            "--criteria", CRITERIA,
            "--resolve-by", "2026-12-18",
            "--db", str(tmp_path / "sub.db"),
        ],
    )
    assert (tmp_path / "sub.db").exists()
    assert not (tmp_path / "top.db").exists()


def test_json_output_is_machine_readable(tmp_path, monkeypatch, capsys):
    import json

    run(
        monkeypatch,
        [
            "forecast",
            "--text", TEXT,
            "--criteria", CRITERIA,
            "--resolve-by", "2026-12-18",
            "--db", str(tmp_path / "c.db"),
            "--audit-dir", str(tmp_path / "runs3"),
            "--json",
        ],
    )
    payload = json.loads(capsys.readouterr().out)
    assert 0.0 < payload["probability"] < 1.0
    assert payload["lower"] <= payload["probability"] <= payload["upper"]


def test_simulate_reports_convergence(tmp_path, monkeypatch, capsys):
    import json

    run(monkeypatch, ["simulate", "--agents", "20", "--rounds", "20", "--seed", "3"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["truth"] == 0.7
    assert 0.0 < payload["final_mean"] < 1.0
    assert payload["final_spread"] >= 0.0
    assert "converged_round" in payload


def test_serve_options_are_exported_to_the_environment(tmp_path, monkeypatch):
    from polycast.cli import export_env

    monkeypatch.delenv("POLYCAST_DB", raising=False)
    monkeypatch.delenv("POLYCAST_AUDIT_DIR", raising=False)

    class Args:
        corpus = None
        db = str(tmp_path / "served.db")
        audit_dir = str(tmp_path / "served_runs")
        model = None

    export_env(Args())
    assert os.environ["POLYCAST_DB"] == str(tmp_path / "served.db")
    assert os.environ["POLYCAST_AUDIT_DIR"] == str(tmp_path / "served_runs")


def test_export_env_leaves_unset_options_alone(monkeypatch):
    from polycast.cli import export_env

    monkeypatch.delenv("POLYCAST_MODEL", raising=False)

    class Args:
        corpus = None
        db = None
        audit_dir = None
        model = None

    export_env(Args())
    assert "POLYCAST_MODEL" not in os.environ
