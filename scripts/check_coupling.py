"""Compare uncoupled/coupled references after initial phase offsets and a phase disturbance."""

import argparse
import csv
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/codyn_isaac/codyn_isaac"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "log/codyn-isaac/reference_coupling")
    args = parser.parse_args()
    torch.set_num_threads(1)
    package = types.ModuleType("coupling_check")
    package.__path__ = [str(PACKAGE / "tasks/manager_based/codyn_isaac/mdp")]
    sys.modules[package.__name__] = package
    GaitReference = importlib.import_module("coupling_check.reference").GaitReference
    base_speed = torch.tensor([0.8, 1.2, 1.6, 0.8, 1.2, 1.6, 1.0, 1.4], device=args.device)
    results = {}
    for gain in (0.0, 0.25):
        torch.manual_seed(42)
        reference = GaitReference(PACKAGE / "assets/oscillator/M01_left.pt", len(base_speed), args.device, coupling_gain=gain)
        reference.speed.copy_(base_speed)
        bank = reference._bank_at(base_speed)
        count = bank.shape[1]
        left = torch.full((len(base_speed),), 25, device=args.device, dtype=torch.long)
        offset = torch.tensor([40, 40, 40, -40, -40, -40, 40, -40], device=args.device)
        index = torch.stack([left, (left + count // 2 + offset) % count], dim=1)
        reference.state[:] = bank[torch.arange(len(base_speed), device=args.device)[:, None], index]
        reference._refresh(torch.arange(len(base_speed), device=args.device))
        errors = []
        for step in range(801):
            seconds = step * 0.02
            if step == 300:
                # Move only the right leg forward by 20% of a cycle, preserving the learned orbit.
                bank = reference._bank_at(reference.speed)
                phases = reference._phases(reference.state, bank)
                right = (torch.round(phases[:, 1] * count / (2 * torch.pi)).long() + 40) % count
                reference.state[:, 1] = bank[torch.arange(len(base_speed), device=args.device), right]
                reference._refresh(torch.arange(len(base_speed), device=args.device))
            errors.append(reference.phase_error().cpu().numpy())
            if step < 800:
                speed = base_speed.clone()
                speed[-2:] = 1.2 + 0.4 * torch.sin(torch.tensor(seconds * 2 * np.pi / 8, device=args.device))
                reference.step(speed, 0.02)
        errors = np.rad2deg(np.asarray(errors))
        tail = np.abs(errors[-100:])
        label = "uncoupled" if gain == 0 else "coupled"
        results[label] = {"gain": gain, "initial_error_deg": errors[0].tolist(),
                          "late_mean_abs_error_deg": tail.mean(0).tolist(), "late_max_abs_error_deg": float(tail.max())}
        results[label]["errors"] = errors
        print(label, "last 2s mean absolute phase error (deg):", np.round(tail.mean(0), 3).tolist(), flush=True)
    assert np.abs(results["uncoupled"]["errors"][200:290, :6]).mean() > 40
    assert np.abs(results["coupled"]["errors"][200:290]).max() < 5
    assert results["coupled"]["late_max_abs_error_deg"] < 5
    # Opposite rate corrections conserve mean progression and never reverse time.
    probe = torch.tensor([[[0., 0.], [0., 0.]]], device=args.device)
    bank = reference._bank_at(base_speed[:1])
    probe[:, 0] = bank[:, 25]; probe[:, 1] = bank[:, 65]
    rates = reference._coupling_rates(probe, bank)
    torch.testing.assert_close(rates.sum(1), torch.tensor([2.], device=args.device))
    assert rates.min() >= 0.75 and rates.max() <= 1.25
    args.output.mkdir(parents=True, exist_ok=True)
    times = np.arange(801) * 0.02
    with (args.output / "phase_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["time_s", "coupling_gain", "case", "phase_error_deg"])
        for label in results:
            for i, row in enumerate(results[label]["errors"]):
                writer.writerows([times[i], results[label]["gain"], j, value] for j, value in enumerate(row))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for label, title, ax in zip(("uncoupled", "coupled"), ("无耦合：初始相位差与扰动后误差", "反相耦合：恢复半周期交替"), axes):
        errors = results[label]["errors"]
        ax.plot(times, errors, lw=0.8, alpha=0.6)
        ax.axhline(0, color="#333333", lw=1, linestyle="--")
        ax.axvline(6, color="#d45528", linestyle=":", label="右腿相位扰动")
        ax.set(xlabel="时间（秒）", ylabel="相对180°相位差的误差（度）", title=title, ylim=(-180, 180))
        ax.grid(alpha=0.2); ax.legend()
    fig.savefig(args.output / "phase_coordination.png", dpi=180); plt.close(fig)
    for result in results.values():
        result.pop("errors")
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("PASS: recovery from +/-72-degree offsets and a phase disturbance; constant and changing speeds; bounded forward progression.", flush=True)


if __name__ == "__main__":
    main()
