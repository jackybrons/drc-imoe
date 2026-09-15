"""Paper experiment matrix, real-data smoke checks, and four-GPU execution.

``formal-commands`` only prints jobs; ``formal-run`` launches independent GPU
queues. ``smoke`` performs one tiny step per family to validate the data path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.loader import BatteryDataset, BatteryDataset1, BatteryDataset2
from experiments.main.run import build_parser as build_run_parser, build_setting


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "reproduction" / "experiment_smoke"
SEEDS = (2025, 2026, 2027)


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    question: str
    datasets: str
    baselines: str
    metrics: str
    positive_criterion: str


SECTIONS = (
    ExperimentSection("problem", "Can shared dual routing use statistical and curve information within one model?", "fixed battery splits across five data families", "published predictors", "protocol audit", "the test set is not used for selection"),
    ExperimentSection("data", "Does the evaluation cover different chemistries and operating conditions?", "UL-NCA/NCM/NCMNCA, TPSL, LSD", "n/a", "battery/window statistics", "audit and smoke checks completed for all five families"),
    ExperimentSection("main", "Does DRC-iMOE outperform single-router models and published baselines?", "all five families", "iMOE, iMOE-CSR, dual_router full-fusion comparison, published predictors", "RMSE/MAE/MAPE/R2", "report three-seed mean, standard deviation, and paired win rate"),
    ExperimentSection("generalization", "Does the advantage hold across operating conditions?", "TPSL Arbitrary->Fixed and Fixed->Arbitrary", "same baselines", "target RMSE/degradation", "report target-domain RMSE and degradation rate as three-seed paired results"),
    ExperimentSection("data_efficiency", "Is the model more effective with fewer training batteries?", "UL-NCA: 1/2/3; TPSL: 10/19/29/38; LSD: 15/29/43/57 batteries", "iMOE, iMOE-CSR", "RMSE curve/AUC", "report three-seed results using actual battery counts and fixed IDs"),
    ExperimentSection("horizon", "Does the advantage persist over longer horizons?", "pred_len 10/25/50", "iMOE, iMOE-CSR", "RMSE/error slope", "report three-seed results for all three preregistered horizons"),
    ExperimentSection("ablation", "Do curve information, dual routing, learned fusion, noise, and sparse routing help?", "UL-NCA/TPSL/LSD", "stat-only/curve-only/fixed/learned/no-noise/no-top-k", "RMSE/parameters", "evaluate each architectural contribution without selecting configurations on the test set"),
    ExperimentSection("efficiency", "Is the resource cost reasonable?", "three representative families", "iMOE, iMOE-CSR, dual_router", "params/memory/latency/throughput", "report measured values without presupposing the conclusion"),
    ExperimentSection("stability", "Are the results sensitive to initialization?", "all formal runs", "best baseline", "mean/std/win rate", "report three-seed mean, standard deviation, and paired win rate"),
    ExperimentSection("interpretability", "Do attributions correspond to degradation transitions?", "representative trajectories", "occlusion/simple attribution", "attribution/error change", "high-attribution regions should align with transitions and increase error when occluded"),
)


@dataclass(frozen=True)
class DataFamily:
    dataset: str
    condition: str
    pattern: str
    dataset_class: type
    formal_hidden_dim: int


FAMILIES = (
    DataFamily("UL-NCA", "CY25-025_1", "datasets/processed/UL-NCA/CY25-025_1-*.csv", BatteryDataset, 64),
    DataFamily("UL-NCM", "CY25-05_1", "datasets/processed/UL-NCM/CY25-05_1-*.csv", BatteryDataset, 64),
    DataFamily("UL-NCMNCA", "CY25-05_1", "datasets/processed/UL-NCMNCA/CY25-05_1-*.csv", BatteryDataset, 64),
    DataFamily("TPSL", "Arbitrary", "datasets/processed/TPSL-Arbitrary/*/combined_data.csv", BatteryDataset1, 128),
    DataFamily("LSD", "LSD", "datasets/processed/LSD/*.csv", BatteryDataset2, 64),
)
TPSL_FIXED = DataFamily("TPSL", "Fixed", "datasets/processed/TPSL-Fixed/*/combined_data.csv", BatteryDataset1, 128)
BASELINES = (
    "iMOE",
    "iMOE_CSR",
    "ConditionedLSTM",
    "ConditionedMLP",
    "PATCHTST",
    "Informer",
    "DegradationFormer",
)
PAPER_MODELS = ("iMOE_SDR",) + BASELINES
CORE_ROUTER_MODELS = ("iMOE", "iMOE_CSR", "iMOE_SDR")
HORIZONS = (10, 25, 50)
DATA_EFFICIENCY_COUNTS = {
    ("UL-NCA", "CY25-025_1"): (1, 2, 3),
    ("TPSL", "Arbitrary"): (10, 19, 29, 38),
    ("LSD", "LSD"): (15, 29, 43, 57),
}


@dataclass(frozen=True)
class AblationVariant:
    name: str
    model: str
    flags: str = ""


SDR_ABLATIONS = (
    AblationVariant("stat_only", "iMOE"),
    AblationVariant("curve_only", "iMOE_CSR", "--top_k 4"),
    AblationVariant("fixed_fusion", "iMOE_SDR", "--fusion_mode fixed"),
    AblationVariant("learned_fusion", "iMOE_SDR", "--fusion_mode learned"),
    AblationVariant(
        "without_noise",
        "iMOE_SDR",
        "--disable_noisy_routing",
    ),
    AblationVariant(
        "without_top_k",
        "iMOE_SDR",
        "--disable_top_k",
    ),
)
SDR_CONFIG_KEYS = (
    "learning_rate",
    "diverloss",
    "baseline_top_k",
    "csr_top_k",
    "curve_channels",
    "fusion_mode",
)


def load_sdr_configs(summary_paths: Iterable[Path]) -> dict[tuple[str, str], dict]:
    configs = {}
    for path in summary_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("workflow") != "iMOE_SDR":
            raise ValueError(f"{path} is not an iMOE_SDR locked summary")
        try:
            dataset = payload["dataset"]
            condition = payload["condition"]
            config = dict(payload["locked_config_without_seed"])
        except KeyError as exc:
            raise ValueError(f"{path} is missing locked SDR protocol fields") from exc
        missing = [key for key in SDR_CONFIG_KEYS if key not in config]
        if missing:
            raise ValueError(f"{path} locked config is missing {missing}")
        config.setdefault("fusion_gate_bias", 0.0)
        key = (dataset, condition)
        if key in configs:
            raise ValueError(f"duplicate SDR config for {dataset}/{condition}")
        configs[key] = config
    return configs


def _sdr_config_flags(config: Mapping[str, object]) -> str:
    options = (
        ("--learning_rate", config["learning_rate"]),
        ("--diverloss", config["diverloss"]),
        ("--baseline_top_k", config["baseline_top_k"]),
        ("--csr_top_k", config["csr_top_k"]),
        ("--curve_channels", config["curve_channels"]),
        ("--fusion_mode", config["fusion_mode"]),
        ("--fusion_gate_bias", config.get("fusion_gate_bias", 0.0)),
    )
    return " ".join(f"{name} {value}" for name, value in options)


def _sdr_flags_and_protocol(
    family: DataFamily,
    sdr_configs: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[str, str]:
    config = sdr_configs.get((family.dataset, family.condition))
    if config is None:
        return "", "untuned_default"
    return _sdr_config_flags(config), "locked"


def _ablation_flags(
    family: DataFamily,
    variant: AblationVariant,
    sdr_configs: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[str, str, bool]:
    config = sdr_configs.get((family.dataset, family.condition))
    protocol = "locked" if config is not None else "untuned_default"
    if config is None:
        config = {
            "learning_rate": 1e-4,
            "diverloss": 0.5,
            "baseline_top_k": 2,
            "csr_top_k": 4,
            "curve_channels": 4,
            "fusion_mode": "learned",
            "fusion_gate_bias": 0.0,
        }
    if variant.name == "stat_only":
        flags = (
            f"--learning_rate {config['learning_rate']} --diverloss {config['diverloss']} "
            f"--top_k {config['baseline_top_k']}"
        )
        return flags, protocol, False
    if variant.name == "curve_only":
        flags = (
            f"--learning_rate {config['learning_rate']} --diverloss {config['diverloss']} "
            f"--top_k {config['csr_top_k']} --curve_channels {config['curve_channels']}"
        )
        return flags, protocol, False
    matching_main_fusion = (
        variant.name == f"{config['fusion_mode']}_fusion"
    )
    flags = " ".join((_sdr_config_flags(config), variant.flags)).strip()
    return flags, protocol, matching_main_fusion


def model_args(dataset: str, pred_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        seq_len=50,
        pred_len=pred_len,
        enc_in=1,
        hidden_dim=32,
        d_model=32,
        e_layers=2,
        d_layers=1,
        d_ff=64,
        dropout=0.1,
        patch_size=2,
        num_experts=5,
        top_k=2,
        baseline_top_k=2,
        csr_top_k=4,
        curve_channels=4,
        fusion_mode="learned",
        fusion_gate_bias=0.0,
        disable_noisy_routing=False,
        disable_top_k=False,
        disable_trend=False,
        disable_residual=False,
        alpha=10,
        diverloss=0.0,
        soc=20,
    )


def _real_samples(family: DataFamily, pred_len: int, max_batteries: int):
    paths = sorted(ROOT.glob(family.pattern))[:max_batteries]
    if not paths:
        raise FileNotFoundError(f"No real battery files match {family.pattern}")
    samples = []
    used = []
    for path in paths:
        # Only enough rows for one window are parsed; formal loaders remain untouched.
        data = pd.read_csv(path, nrows=pred_len + 2)
        dataset = family.dataset_class(data, window_size=pred_len, soc=20)
        if len(dataset) < 1:
            raise ValueError(f"Battery {path} is too short for smoke pred_len={pred_len}")
        inputs, target = dataset[0]
        samples.append((inputs, target))
        used.append(str(path.relative_to(ROOT)))
    return samples, used


def _load_model(name: str, args: SimpleNamespace) -> torch.nn.Module:
    try:
        module = importlib.import_module(f"models.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"models.{name}":
            raise RuntimeError(
                f"Model {name} is not integrated yet; expected models/{name}.py with Model(args)."
            ) from exc
        raise
    return module.Model(args).float()


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction.detach().cpu().numpy().reshape(-1)
    true = target.detach().cpu().numpy().reshape(-1)
    error = pred - true
    mse = float(np.mean(error**2))
    denominator = np.maximum(np.abs(true), 1e-8)
    ss_total = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "mape_percent": float(np.mean(np.abs(error) / denominator) * 100.0),
        "r2": 1.0 - float(np.sum(error**2)) / ss_total if ss_total > 0 else float("nan"),
    }


def _batch(family: DataFamily, pred_len: int, batteries: int, device: torch.device):
    samples, paths = _real_samples(family, pred_len, batteries)
    batch_inputs, target = next(iter(DataLoader(samples, batch_size=len(samples), shuffle=False)))
    return tuple(value.to(device) for value in batch_inputs), target.to(device), paths


def _prediction(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    output = model(*inputs)
    return output[0] if isinstance(output, tuple) else output


def _run_step(
    model_name: str,
    family: DataFamily,
    pred_len: int,
    batteries: int,
    device: torch.device,
    seed: int = 2025,
    disable_trend: bool = False,
    disable_residual: bool = False,
) -> dict:
    torch.manual_seed(seed)
    inputs, target, paths = _batch(family, pred_len, batteries, device)
    args = model_args(family.dataset, pred_len)
    args.disable_trend = disable_trend
    args.disable_residual = disable_residual
    model = _load_model(model_name, args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    prediction = _prediction(model, inputs)
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"{family.dataset} output shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    metrics = regression_metrics(prediction, target)
    if not all(math.isfinite(value) for key, value in metrics.items() if key != "r2"):
        raise RuntimeError(f"Non-finite smoke metrics for {family.dataset}")
    return {
        "dataset": family.dataset,
        "condition": family.condition,
        "batteries": paths,
        "seed": seed,
        "prediction_shape": list(prediction.shape),
        "backward_loss": float(loss.detach().cpu()),
        "metrics": metrics,
    }


def run_smoke(model_name: str, output_dir: Path, max_batteries: int, pred_len: int, device_name: str) -> dict:
    if max_batteries < 2:
        raise ValueError("max_batteries must be at least 2 to smoke-test data efficiency")
    device = torch.device(device_name)
    sections: dict[str, object] = {}

    sections["problem"] = {
        "status": "passed",
        "protocol": "battery-level split; validation selection; locked test; smoke has no epoch training",
        "model_independent_of_imoe": model_name != "iMOE",
    }

    inventory = []
    for family in FAMILIES:
        samples, paths = _real_samples(family, pred_len, max_batteries)
        inventory.append(
            {
                "dataset": family.dataset,
                "condition": family.condition,
                "batteries": paths,
                "sample_count": len(samples),
                "input_shapes": [list(value.shape) for value in samples[0][0]],
                "target_shape": list(samples[0][1].shape),
            }
        )
    sections["data"] = {"status": "passed", "families": inventory}

    sections["main"] = {
        "status": "passed",
        "runs": [
            _run_step(model_name, family, pred_len, max_batteries, device)
            for family in FAMILIES
        ],
    }

    torch.manual_seed(2025)
    source_inputs, source_target, source_paths = _batch(
        FAMILIES[3], pred_len, max_batteries, device
    )
    target_inputs, target, target_paths = _batch(TPSL_FIXED, pred_len, max_batteries, device)
    transfer_model = _load_model(model_name, model_args("TPSL", pred_len)).to(device)
    transfer_optimizer = torch.optim.SGD(transfer_model.parameters(), lr=1e-4)
    transfer_optimizer.zero_grad()
    source_prediction = _prediction(transfer_model, source_inputs)
    source_loss = torch.nn.functional.mse_loss(source_prediction, source_target)
    source_loss.backward()
    transfer_optimizer.step()
    transfer_model.eval()
    with torch.no_grad():
        target_prediction = _prediction(transfer_model, target_inputs)
    sections["generalization"] = {
        "status": "passed",
        "source": {
            "condition": "Arbitrary",
            "batteries": source_paths,
            "one_step_loss": float(source_loss.detach().cpu()),
        },
        "target": {
            "condition": "Fixed",
            "batteries": target_paths,
            "metrics": regression_metrics(target_prediction, target),
        },
    }

    sections["data_efficiency"] = {
        "status": "passed",
        "runs": [
            _run_step(model_name, FAMILIES[0], pred_len, battery_count, device)
            for battery_count in (1, max_batteries)
        ],
        "battery_counts": [1, max_batteries],
    }

    short_horizon = max(2, pred_len // 2)
    sections["horizon"] = {
        "status": "passed",
        "runs": [
            _run_step(model_name, FAMILIES[0], horizon, max_batteries, device)
            for horizon in (short_horizon, pred_len)
        ],
        "pred_lens": [short_horizon, pred_len],
    }

    ablation_runs = []
    for name, disable_trend, disable_residual in (
        ("full", False, False),
        ("no_trend", True, False),
        ("no_residual", False, True),
    ):
        run = _run_step(
            model_name,
            FAMILIES[3],
            pred_len,
            max_batteries,
            device,
            disable_trend=disable_trend,
            disable_residual=disable_residual,
        )
        run["variant"] = name
        ablation_runs.append(run)
    sections["ablation"] = {"status": "passed", "runs": ablation_runs}

    inputs, _, efficiency_paths = _batch(FAMILIES[0], pred_len, max_batteries, device)
    efficiency_model = _load_model(model_name, model_args("UL-NCA", pred_len)).to(device).eval()
    with torch.no_grad():
        _prediction(efficiency_model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        timed_prediction = _prediction(efficiency_model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    sections["efficiency"] = {
        "status": "passed",
        "batteries": efficiency_paths,
        "parameter_count": sum(parameter.numel() for parameter in efficiency_model.parameters()),
        "one_forward_ms": elapsed_ms,
        "prediction_shape": list(timed_prediction.shape),
        "formal_workflow": "reuse each main-result checkpoint; do not retrain",
    }

    stability_runs = [
        _run_step(model_name, FAMILIES[0], pred_len, max_batteries, device, seed=seed)
        for seed in (2025, 2026)
    ]
    sections["stability"] = {
        "status": "passed",
        "runs": stability_runs,
        "rmse_std": float(np.std([run["metrics"]["rmse"] for run in stability_runs])),
    }

    torch.manual_seed(2025)
    interpretation_inputs, interpretation_target, interpretation_paths = _batch(
        FAMILIES[0], pred_len, max_batteries, device
    )
    curve = interpretation_inputs[0].detach().clone().requires_grad_(True)
    grad_inputs = (curve,) + interpretation_inputs[1:]
    interpretation_model = _load_model(model_name, model_args("UL-NCA", pred_len)).to(device).eval()
    base_prediction = _prediction(interpretation_model, grad_inputs)
    base_mse = torch.nn.functional.mse_loss(base_prediction, interpretation_target)
    base_mse.backward()
    curve_gradient = curve.grad.detach().abs().mean(dim=0)
    top_index = int(torch.argmax(curve_gradient).item())
    occluded_curve = interpretation_inputs[0].detach().clone()
    occluded_curve[:, top_index] = 0.0
    with torch.no_grad():
        occluded_prediction = _prediction(
            interpretation_model, (occluded_curve,) + interpretation_inputs[1:]
        )
        occluded_mse = torch.nn.functional.mse_loss(occluded_prediction, interpretation_target)
    sections["interpretability"] = {
        "status": "passed",
        "batteries": interpretation_paths,
        "input_gradient_l1": float(curve_gradient.sum().cpu()),
        "top_curve_index": top_index,
        "base_mse": float(base_mse.detach().cpu()),
        "occluded_mse": float(occluded_mse.detach().cpu()),
        "occluded_mse_delta": float((occluded_mse - base_mse.detach()).cpu()),
        "formal_workflow": "reuse each main-result checkpoint; do not retrain",
    }

    payload = {
        "status": "passed",
        "scope": "pipeline smoke only; not a paper result",
        "model": model_name,
        "pred_len": pred_len,
        "sections": sections,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "smoke_results.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["result_file"] = str(path)
    return payload


def _model_flags(model: str) -> str:
    if model == "iMOE_CSR":
        return "--top_k 4"
    if model == "iMOE_SDR":
        return "--fusion_mode learned"
    return ""


def _formal_command(
    common: str,
    model: str,
    family: DataFamily,
    seed: int,
    description: str,
    *,
    pred_len: int = 50,
    dataaccess: int = 100,
    train_battery_count: int | None = None,
    extra_flags: str = "",
    test_condition: str | None = None,
) -> str:
    default_flags = _model_flags(model)
    if model == "iMOE_SDR" and "--fusion_mode" in extra_flags:
        default_flags = ""
    if model == "iMOE_CSR" and "--top_k" in extra_flags:
        default_flags = ""
    flags = " ".join(value for value in (default_flags, extra_flags) if value)
    transfer = f" --test_condition {test_condition}" if test_condition else ""
    subset = (
        f" --train_battery_count {train_battery_count}"
        if train_battery_count is not None
        else ""
    )
    suffix = f" {flags}" if flags else ""
    return (
        f"python -m experiments.main.run {common} --model {model} --dataset {family.dataset} "
        f"--condition {family.condition}{transfer} --hidden_dim {family.formal_hidden_dim} "
        f"--pred_len {pred_len} --dataaccess {dataaccess}{subset} --seed {seed}{suffix} "
        f"--des {description}"
    )


def formal_commands(
    model: str = "iMOE_SDR",
    sdr_configs: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
) -> Iterable[str]:
    """Yield locked formal runs without launching them.

    The fixed/learned fusion variant matching the selected SDR configuration
    reuses its main-result checkpoint.  Other ablations are emitted explicitly
    so they inherit the same selected optimization and routing configuration.
    """
    common = (
        "--is_training 1 --train_epochs 1500 --patience 250 "
        "--batch_size 32 --eval_batch_size 64 "
        "--data_root datasets/processed "
        "--learning_rate 0.0001 --itr 1"
    )
    sdr_configs = {} if sdr_configs is None else sdr_configs
    comparison_models = PAPER_MODELS
    if model not in comparison_models:
        comparison_models = (model,) + comparison_models
    for family in FAMILIES:
        for comparison_model in comparison_models:
            for seed in SEEDS:
                sdr_flags, protocol = _sdr_flags_and_protocol(family, sdr_configs)
                extra_flags = sdr_flags if comparison_model == "iMOE_SDR" else ""
                prefix = protocol if comparison_model == "iMOE_SDR" else "locked"
                yield _formal_command(
                    common,
                    comparison_model,
                    family,
                    seed,
                    f"{prefix}_main_{comparison_model.lower()}_{family.dataset.lower()}_s{seed}",
                    extra_flags=extra_flags,
                )
    for comparison_model in comparison_models:
        for seed in SEEDS:
            source_flags, source_protocol = _sdr_flags_and_protocol(FAMILIES[3], sdr_configs)
            fixed_flags, fixed_protocol = _sdr_flags_and_protocol(TPSL_FIXED, sdr_configs)
            yield _formal_command(
                common,
                comparison_model,
                FAMILIES[3],
                seed,
                f"{source_protocol if comparison_model == 'iMOE_SDR' else 'locked'}_generalization_{comparison_model.lower()}_arbitrary_to_fixed_s{seed}",
                extra_flags=source_flags if comparison_model == "iMOE_SDR" else "",
                test_condition="Fixed",
            )
            yield _formal_command(
                common,
                comparison_model,
                TPSL_FIXED,
                seed,
                f"{fixed_protocol if comparison_model == 'iMOE_SDR' else 'locked'}_generalization_{comparison_model.lower()}_fixed_to_arbitrary_s{seed}",
                extra_flags=fixed_flags if comparison_model == "iMOE_SDR" else "",
                test_condition="Arbitrary",
            )
    for family in (FAMILIES[0], FAMILIES[3], FAMILIES[4]):
        full_battery_count = DATA_EFFICIENCY_COUNTS[(family.dataset, family.condition)][-1]
        for train_battery_count in DATA_EFFICIENCY_COUNTS[(family.dataset, family.condition)]:
            if train_battery_count == full_battery_count:
                continue
            for comparison_model in CORE_ROUTER_MODELS:
                for seed in SEEDS:
                    sdr_flags, protocol = _sdr_flags_and_protocol(family, sdr_configs)
                    yield _formal_command(
                        common,
                        comparison_model,
                        family,
                        seed,
                        f"{protocol if comparison_model == 'iMOE_SDR' else 'locked'}_n{train_battery_count}_{comparison_model.lower()}_{family.dataset.lower()}_s{seed}",
                        train_battery_count=train_battery_count,
                        extra_flags=sdr_flags if comparison_model == "iMOE_SDR" else "",
                    )
        for pred_len in HORIZONS:
            if pred_len == 50:
                continue
            for comparison_model in CORE_ROUTER_MODELS:
                for seed in SEEDS:
                    sdr_flags, protocol = _sdr_flags_and_protocol(family, sdr_configs)
                    yield _formal_command(
                        common,
                        comparison_model,
                        family,
                        seed,
                        f"{protocol if comparison_model == 'iMOE_SDR' else 'locked'}_h{pred_len}_{comparison_model.lower()}_{family.dataset.lower()}_s{seed}",
                        pred_len=pred_len,
                        extra_flags=sdr_flags if comparison_model == "iMOE_SDR" else "",
                    )
        for variant in SDR_ABLATIONS:
            variant_flags, protocol, reuse_main = _ablation_flags(
                family, variant, sdr_configs
            )
            if reuse_main:
                continue
            for seed in SEEDS:
                yield _formal_command(
                    common,
                    variant.model,
                    family,
                    seed,
                    f"{protocol}_ablation_{variant.name}_{family.dataset.lower()}_s{seed}",
                    extra_flags=variant_flags,
                )


def _description_from_argv(argv: list[str]) -> str:
    try:
        return argv[argv.index("--des") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("formal command is missing a --des job identifier") from exc


GLOBAL_METRIC_KEYS = ("rmse", "mae", "mape_percent", "r2")


def _run_args_from_argv(argv: list[str]) -> argparse.Namespace:
    argument_start = 3 if len(argv) >= 3 and argv[1] == "-m" else 2
    return build_run_parser().parse_args(argv[argument_start:])


def _metrics_path_from_argv(argv: list[str]) -> Path:
    args = _run_args_from_argv(argv)
    setting = build_setting(args, 0)
    return (
        Path(args.results_dir)
        / args.model
        / args.dataset
        / args.condition
        / setting
        / "metrics.json"
    ).resolve()


def _load_global_metrics(metrics_path: Path) -> dict[str, float]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("global")
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json is missing the global metrics object")
    missing = [key for key in GLOBAL_METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"metrics.json is missing global metrics: {missing}")
    values = {key: float(metrics[key]) for key in GLOBAL_METRIC_KEYS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("metrics.json contains non-finite global metrics")
    return values


def _job_identity(argv: list[str]) -> dict[str, object]:
    args = _run_args_from_argv(argv)
    return {
        "model": args.model,
        "dataset": args.dataset,
        "condition": args.condition,
        "seed": args.seed,
    }


def _saved_marker_record(saved: dict, command: list[str]) -> dict | None:
    if saved.get("command") != command or saved.get("status") != "completed":
        return None
    metrics_path_value = saved.get("metrics_path")
    metrics = saved.get("metrics")
    if not metrics_path_value or not Path(metrics_path_value).is_file():
        return None
    if not isinstance(metrics, dict) or set(metrics) != set(GLOBAL_METRIC_KEYS):
        return None
    try:
        numeric_metrics = {key: float(metrics[key]) for key in GLOBAL_METRIC_KEYS}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in numeric_metrics.values()):
        return None
    return {
        **{key: saved[key] for key in ("model", "dataset", "condition", "seed")},
        "metrics_path": metrics_path_value,
        "metrics": numeric_metrics,
    }


def _run_gpu_queue(
    gpu: int,
    jobs: list[tuple[int, str]],
    output_dir: Path,
    checkpoint_dir: Path,
) -> list[dict]:
    marker_dir = output_dir / "markers"
    log_dir = output_dir / "logs"
    result_artifact_dir = output_dir / "outputs"
    marker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_artifact_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, command in jobs:
        argv = shlex.split(command)
        argv[0] = sys.executable
        argv.extend((
            "--gpu", "0",
            "--checkpoints", str(checkpoint_dir),
            "--results_dir", str(result_artifact_dir),
        ))
        description = _description_from_argv(argv)
        marker_path = marker_dir / f"{description}.done.json"
        marker_payload = {"command": argv[1:], **_job_identity(argv)}
        if marker_path.exists():
            try:
                saved = json.loads(marker_path.read_text(encoding="utf-8"))
                saved_record = _saved_marker_record(saved, marker_payload["command"])
            except (OSError, json.JSONDecodeError, KeyError):
                saved_record = None
            if saved_record is not None:
                print(f"[GPU {gpu}] skip completed {description}", flush=True)
                records.append({
                    "index": index,
                    "description": description,
                    "status": "skipped",
                    **saved_record,
                })
                continue

        expected_metrics_path = _metrics_path_from_argv(argv)
        metrics_state_before = (
            (expected_metrics_path.stat().st_mtime_ns, expected_metrics_path.stat().st_size)
            if expected_metrics_path.is_file()
            else None
        )
        log_path = log_dir / f"{description}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[GPU {gpu}] start {description}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        error = None
        global_metrics = None
        metrics_path = None
        if result.returncode == 0:
            try:
                metrics_path = expected_metrics_path
                if not metrics_path.is_file():
                    raise ValueError("metrics.json was not regenerated by this job")
                if metrics_state_before is not None and metrics_path.is_file():
                    metrics_state_after = (metrics_path.stat().st_mtime_ns, metrics_path.stat().st_size)
                    if metrics_state_after == metrics_state_before:
                        raise ValueError("metrics.json was not regenerated by this job")
                global_metrics = _load_global_metrics(metrics_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
        status = "completed" if result.returncode == 0 and global_metrics is not None else "failed"
        if status == "completed":
            marker_payload.update({
                "gpu": gpu,
                "log": str(log_path),
                "status": "completed",
                "metrics_path": str(metrics_path),
                "metrics": global_metrics,
            })
            marker_path.write_text(
                json.dumps(marker_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[GPU {gpu}] {status} {description}", flush=True)
        records.append(
            {
                "index": index,
                "description": description,
                "status": status,
                "returncode": result.returncode,
                "log": str(log_path),
                **_job_identity(argv),
                "metrics_path": str(metrics_path) if metrics_path is not None else None,
                "metrics": global_metrics,
                **({"error": error} if error else {}),
            }
        )
    return records


def run_formal_jobs(
    model: str,
    gpus: list[int],
    output_dir: Path,
    sdr_configs: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    output_dir = Path(output_dir).resolve()
    checkpoint_dir = Path(
        checkpoint_dir
        if checkpoint_dir is not None
        else ROOT / "checkpoints" / "main" / "formal_matrix"
    ).resolve()
    if not gpus:
        raise ValueError("at least one GPU is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU ids must be unique")
    queues = [[] for _ in gpus]
    commands = list(formal_commands(model, sdr_configs))
    for index, command in enumerate(commands):
        queues[index % len(gpus)].append((index, command))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_run_gpu_queue, gpu, jobs, output_dir, checkpoint_dir)
            for gpu, jobs in zip(gpus, queues)
        ]
        records = [record for future in futures for record in future.result()]
    records.sort(key=lambda record: record["index"])
    summary = {
        "status": "failed" if any(record["status"] == "failed" for record in records) else "completed",
        "gpus": gpus,
        "total_jobs": len(records),
        "completed_jobs": sum(record["status"] == "completed" for record in records),
        "skipped_jobs": sum(record["status"] == "skipped" for record in records),
        "failed_jobs": sum(record["status"] == "failed" for record in records),
        "jobs": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRC-iMOE paper experiment framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="show paper sections and acceptance criteria")
    list_parser.add_argument("--json", action="store_true")
    smoke = subparsers.add_parser("smoke", help="run tiny real-battery pipeline checks")
    smoke.add_argument("--model", default="DegradationFormer")
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    smoke.add_argument("--max-batteries", type=int, default=2)
    smoke.add_argument("--pred-len", type=int, default=10)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    commands = subparsers.add_parser("formal-commands", help="print commands without running training")
    commands.add_argument("--model", default="iMOE_SDR", help="optional additional model")
    commands.add_argument("--num-shards", type=int, default=1)
    commands.add_argument("--shard-index", type=int, default=0)
    commands.add_argument("--sdr-summary", type=Path, action="append", default=[])
    formal_run = subparsers.add_parser("formal-run", help="run the formal matrix across GPU queues")
    formal_run.add_argument("--model", default="iMOE_SDR", help="optional additional model")
    formal_run.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    formal_run.add_argument("--sdr-summary", type=Path, action="append", default=[])
    formal_run.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "main" / "formal_matrix",
    )
    formal_run.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "main" / "formal_matrix",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        rows = [asdict(section) for section in SECTIONS]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for section in SECTIONS:
                print(f"[{section.name}] {section.question}")
                print(f"  datasets: {section.datasets}")
                print(f"  baselines: {section.baselines}")
                print(f"  metrics: {section.metrics}")
                print(f"  positive: {section.positive_criterion}")
        return 0
    if args.command == "smoke":
        payload = run_smoke(args.model, args.output_dir, args.max_batteries, args.pred_len, args.device)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "formal-run":
        configs = load_sdr_configs(args.sdr_summary)
        summary = run_formal_jobs(
            args.model,
            args.gpus,
            args.output_dir,
            configs,
            args.checkpoint_dir,
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "jobs"}, indent=2))
        return 1 if summary["failed_jobs"] else 0
    if args.num_shards < 1:
        raise ValueError("num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    configs = load_sdr_configs(args.sdr_summary)
    for index, command in enumerate(formal_commands(args.model, configs)):
        if index % args.num_shards == args.shard_index:
            print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
