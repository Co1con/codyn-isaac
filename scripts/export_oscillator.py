"""Export a portable inference checkpoint and model-generated phase bank; no Isaac Sim required."""

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source" / "codyn_isaac" / "codyn_isaac"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PACKAGE / "assets" / "oscillator" / "M01_left.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("oscillator_inference", PACKAGE / "tasks/manager_based/codyn_isaac/mdp/oscillator.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = module.SpeedOscillator(checkpoint["geometry"], bounded_compensation=checkpoint.get("version", 1) == 1)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    trained = [r["speed"] for r in checkpoint["data"] if not r["held_out"]]
    speeds = torch.linspace(min(trained), max(trained), 17)
    with torch.no_grad():
        center, axes, basis, frequency, _ = model.parameters_at(speeds)
        initial = center + torch.bmm(torch.stack([axes[:, 0], torch.zeros_like(axes[:, 0])], 1)[:, None], basis.transpose(1, 2)).squeeze(1)
        trajectories = model.trajectory(initial, speeds, 12 / frequency, 1200, 2).numpy()
    bank = []
    for trajectory in trajectories:
        hip = trajectory[:, 0]
        peaks, _ = find_peaks(hip, distance=50)
        selected = peaks[-2:]
        offset = 0.5 * (hip[selected - 1] - hip[selected + 1]) / (hip[selected - 1] - 2 * hip[selected] + hip[selected + 1])
        left, right = selected + offset
        samples = np.linspace(left, right, 201)
        bank.append(np.column_stack([np.interp(samples, np.arange(len(trajectory)), trajectory[:, j]) for j in range(2)]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": checkpoint.get("version", 1), "geometry": checkpoint["geometry"],
                "model_state": checkpoint["model_state"], "phase_bank": torch.tensor(np.array(bank), dtype=torch.float32),
                "phase_speeds": speeds, "subject": checkpoint["arguments"]["subject"], "side": checkpoint["arguments"]["side"],
                "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}, args.output)
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
