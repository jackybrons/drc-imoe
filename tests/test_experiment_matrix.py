import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

IMOEDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMOEDIR))

from experiments.main import experiment_matrix


def test_matrix_contains_all_standard_paper_sections():
    names = {section.name for section in experiment_matrix.SECTIONS}
    assert names == {
        "problem",
        "data",
        "main",
        "generalization",
        "data_efficiency",
        "horizon",
        "ablation",
        "efficiency",
        "stability",
        "interpretability",
    }


def test_formal_commands_are_print_only_and_cover_core_axes():
    commands = list(experiment_matrix.formal_commands())
    assert commands
    assert all(command.startswith("python -m experiments.main.run ") for command in commands)
    assert all("--seed " in command for command in commands)
    assert experiment_matrix.SEEDS == (2025, 2026, 2027)
    for model in experiment_matrix.PAPER_MODELS:
        assert any(f"--model {model} " in command for command in commands)
    for dataset in ("UL-NCA", "UL-NCM", "UL-NCMNCA", "TPSL", "LSD"):
        assert any(f"--dataset {dataset} " in command for command in commands)
    assert experiment_matrix.DATA_EFFICIENCY_COUNTS == {
        ("UL-NCA", "CY25-025_1"): (1, 2, 3),
        ("TPSL", "Arbitrary"): (10, 19, 29, 38),
        ("LSD", "LSD"): (15, 29, 43, 57),
    }
    assert experiment_matrix.HORIZONS == (10, 25, 50)
    for comparison_model in experiment_matrix.CORE_ROUTER_MODELS:
        assert any(
            f"--model {comparison_model} " in command and "--train_battery_count " in command
            for command in commands
        )
        assert any(
            f"--model {comparison_model} " in command and "--pred_len 10" in command
            for command in commands
        )
    assert any("--fusion_mode fixed" in command for command in commands)
    assert any("--disable_noisy_routing" in command for command in commands)
    assert any("--disable_top_k" in command for command in commands)
    assert any(
        "--dataset TPSL --condition Arbitrary --test_condition Fixed" in command
        for command in commands
    )
    assert any(
        "--dataset TPSL --condition Fixed --test_condition Arbitrary" in command
        for command in commands
    )
    assert all(
        "--hidden_dim 128" in command
        for command in commands
        if "--dataset TPSL " in command
    )
    assert all(
        "--hidden_dim 64" in command
        for command in commands
        if "--dataset UL-NCA " in command
    )
    assert len(commands) == 339
    assert not any("subprocess" in command or "Start-Process" in command for command in commands)


def test_sdr_ablation_matrix_has_six_scientific_comparisons():
    variants = {variant.name: variant for variant in experiment_matrix.SDR_ABLATIONS}
    assert set(variants) == {
        "stat_only",
        "curve_only",
        "fixed_fusion",
        "learned_fusion",
        "without_noise",
        "without_top_k",
    }
    assert variants["stat_only"].model == "iMOE"
    assert variants["curve_only"].model == "iMOE_CSR"
    assert variants["fixed_fusion"].flags == "--fusion_mode fixed"
    assert variants["learned_fusion"].flags == "--fusion_mode learned"
    assert variants["without_noise"].flags == "--disable_noisy_routing"
    assert variants["without_top_k"].flags == "--disable_top_k"


def test_locked_sdr_summary_is_injected_into_main_and_ablation_commands(tmp_path):
    summary_path = tmp_path / "locked.json"
    summary_path.write_text(json.dumps({
        "workflow": "iMOE_SDR",
        "dataset": "UL-NCA",
        "condition": "CY25-025_1",
        "locked_config_without_seed": {
            "learning_rate": 0.0003,
            "diverloss": 0.1,
            "baseline_top_k": 1,
            "csr_top_k": 3,
            "curve_channels": 8,
            "fusion_mode": "learned",
        },
    }), encoding="utf-8")
    configs = experiment_matrix.load_sdr_configs([summary_path])
    commands = list(experiment_matrix.formal_commands(sdr_configs=configs))

    main = next(
        command for command in commands
        if "--model iMOE_SDR " in command
        and "--dataset UL-NCA " in command
        and "--des locked_main_" in command
    )
    for option in (
        "--learning_rate 0.0003",
        "--diverloss 0.1",
        "--baseline_top_k 1",
        "--csr_top_k 3",
        "--curve_channels 8",
        "--fusion_mode learned",
        "--fusion_gate_bias 0.0",
    ):
        assert option in main
    stat_only = next(
        command for command in commands
        if "--des locked_ablation_stat_only_ul-nca" in command
    )
    assert "--model iMOE " in stat_only
    assert "--learning_rate 0.0003" in stat_only
    assert "--diverloss 0.1" in stat_only
    assert "--top_k 1" in stat_only
    assert not any(
        "locked_ablation_learned_fusion_ul-nca" in command
        for command in commands
    )


def test_formal_run_uses_serial_gpu_queues_and_resume_markers(tmp_path, monkeypatch):
    commands = tuple(
        f"python -m experiments.main.run --model iMOE_SDR --des job_{index}"
        for index in range(4)
    )
    monkeypatch.setattr(experiment_matrix, "formal_commands", lambda model, configs: commands)
    calls = []
    failed_once = set()

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        description = argv[argv.index("--des") + 1]
        if description == "job_2" and description not in failed_once:
            failed_once.add(description)
            return SimpleNamespace(returncode=1)
        metrics_path = experiment_matrix._metrics_path_from_argv(argv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({
            "global": {
                "rmse": 0.1,
                "mae": 0.08,
                "mape_percent": 4.0,
                "r2": 0.9,
            },
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment_matrix.subprocess, "run", fake_run)
    first = experiment_matrix.run_formal_jobs("iMOE_SDR", [0, 1], tmp_path)

    assert first["completed_jobs"] == 3
    assert first["failed_jobs"] == 1
    assert len(list((tmp_path / "markers").glob("*.done.json"))) == 3
    second = experiment_matrix.run_formal_jobs("iMOE_SDR", [0, 1], tmp_path)
    assert second["completed_jobs"] == 1
    assert second["skipped_jobs"] == 3
    assert len(calls) == 5
    assert all(record["metrics_path"] for record in second["jobs"])
    assert all(record["metrics"]["rmse"] == 0.1 for record in second["jobs"])
    assert all(
        {"model", "dataset", "condition", "seed"} <= set(record)
        for record in second["jobs"]
    )
    gpu_by_job = {
        argv[argv.index("--des") + 1]: gpu
        for argv, gpu in calls
    }
    assert gpu_by_job == {"job_0": "0", "job_1": "1", "job_2": "0", "job_3": "1"}
    assert len(list((tmp_path / "markers").glob("*.done.json"))) == 4


def test_formal_run_rejects_success_without_metrics_artifact(tmp_path, monkeypatch):
    commands = ("python -m experiments.main.run --model iMOE_SDR --des missing_metrics",)
    monkeypatch.setattr(experiment_matrix, "formal_commands", lambda model, configs: commands)
    monkeypatch.setattr(
        experiment_matrix.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=0),
    )
    argv = [
        sys.executable,
        "-m",
        "experiments.main.run",
        "--model",
        "iMOE_SDR",
        "--des",
        "missing_metrics",
        "--gpu",
        "0",
        "--checkpoints",
        str((tmp_path / "checkpoints").resolve()),
    ]
    stale_metrics = experiment_matrix._metrics_path_from_argv(argv)
    stale_metrics.parent.mkdir(parents=True, exist_ok=True)
    stale_metrics.write_text(json.dumps({
        "global": {"rmse": 0.1, "mae": 0.1, "mape_percent": 1.0, "r2": 0.9},
    }), encoding="utf-8")

    summary = experiment_matrix.run_formal_jobs("iMOE_SDR", [0], tmp_path)

    assert summary["failed_jobs"] == 1
    assert summary["jobs"][0]["metrics"] is None
    assert "not regenerated" in summary["jobs"][0]["error"]
    assert not list((tmp_path / "markers").glob("*.done.json"))


def test_real_data_smoke_loads_each_family_and_writes_results(tmp_path):
    payload = experiment_matrix.run_smoke(
        model_name="ConditionedMLP",
        output_dir=tmp_path,
        max_batteries=2,
        pred_len=10,
        device_name="cpu",
    )
    assert payload["status"] == "passed"
    assert set(payload["sections"]) == {
        "problem",
        "data",
        "main",
        "generalization",
        "data_efficiency",
        "horizon",
        "ablation",
        "efficiency",
        "stability",
        "interpretability",
    }
    assert {row["dataset"] for row in payload["sections"]["main"]["runs"]} == {
        "UL-NCA",
        "UL-NCM",
        "UL-NCMNCA",
        "TPSL",
        "LSD",
    }
    transfer = payload["sections"]["generalization"]
    assert transfer["source"]["condition"] == "Arbitrary"
    assert transfer["target"]["condition"] == "Fixed"
    assert len(transfer["source"]["batteries"]) == 2
    assert len(transfer["target"]["batteries"]) == 2
    assert payload["sections"]["data_efficiency"]["battery_counts"] == [1, 2]
    assert len(payload["sections"]["horizon"]["pred_lens"]) == 2
    assert {run["variant"] for run in payload["sections"]["ablation"]["runs"]} == {
        "full", "no_trend", "no_residual"
    }
    assert payload["sections"]["efficiency"]["parameter_count"] > 0
    assert {run["seed"] for run in payload["sections"]["stability"]["runs"]} == {2025, 2026}
    assert payload["sections"]["interpretability"]["input_gradient_l1"] >= 0
    stored = json.loads((tmp_path / "smoke_results.json").read_text(encoding="utf-8"))
    assert stored["scope"] == "pipeline smoke only; not a paper result"


def test_cli_list_does_not_require_model_integration():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.main.experiment_matrix", "list"],
        cwd=IMOEDIR,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "[main]" in result.stdout
    assert "[interpretability]" in result.stdout
