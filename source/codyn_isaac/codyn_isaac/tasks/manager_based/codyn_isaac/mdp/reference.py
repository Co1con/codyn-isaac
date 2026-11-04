"""Frozen hip/knee oscillators with bidirectional anti-phase coordination."""

import torch

from .oscillator import SpeedOscillator

JOINT_NAMES = ("left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint")


class GaitReference:
    def __init__(self, checkpoint, num_envs, device, substeps=4, coupling_gain=0.25):
        data = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model = SpeedOscillator(data["geometry"], bounded_compensation=data["version"] == 1).to(device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval().requires_grad_(False)
        self.bank = data["phase_bank"].to(device)[:, :-1]
        self.bank_speeds = data["phase_speeds"].to(device)
        self.num_envs, self.device, self.substeps = num_envs, device, substeps
        self.coupling_gain = coupling_gain
        if not 0 <= coupling_gain < 1:
            raise ValueError("coupling_gain must be in [0, 1) to keep both oscillators moving forward")
        self.state = torch.empty(num_envs, 2, 2, device=device)
        self.speed = torch.ones(num_envs, device=device)
        self.position = torch.empty(num_envs, 4, device=device)
        self.velocity = torch.empty_like(self.position)
        self.signs = torch.tensor([-1., 1., -1., 1.], device=device)
        self.reset(torch.arange(num_envs, device=device), self.speed.clone())

    def _bank_at(self, speed):
        # 周期表用于初始化和时间相位估计；原始动力学仍直接使用命令速度。
        speed = speed.clamp(self.bank_speeds[0], self.bank_speeds[-1])
        high = torch.searchsorted(self.bank_speeds, speed.contiguous()).clamp(1, len(self.bank_speeds) - 1)
        low = high - 1
        weight = (speed - self.bank_speeds[low]) / (self.bank_speeds[high] - self.bank_speeds[low])
        return torch.lerp(self.bank[low], self.bank[high], weight[:, None, None])

    @staticmethod
    def _phases(state, bank):
        """Project onto the model cycle's nearest segment; bank samples are uniform in time."""
        scale = bank.std(dim=1, unbiased=False).clamp_min(1e-6)
        points = bank / scale[:, None]
        segments = points.roll(-1, dims=1) - points
        offset = state[:, :, None] / scale[:, None, None] - points[:, None]
        fraction = ((offset * segments[:, None]).sum(-1) / segments.square().sum(-1)[:, None].clamp_min(1e-12)).clamp(0, 1)
        distance = (offset - fraction[..., None] * segments[:, None]).square().sum(-1)
        index = distance.argmin(dim=-1)
        fraction = fraction.gather(-1, index[..., None]).squeeze(-1)
        return (index + fraction) * (2 * torch.pi / bank.shape[1])

    def phase_error(self):
        """Signed deviation from a half-cycle left/right difference, in [-pi, pi]."""
        phases = self._phases(self.state, self._bank_at(self.speed))
        error = phases[:, 1] - phases[:, 0] - torch.pi
        return torch.atan2(error.sin(), error.cos())

    def _coupling_rates(self, state, bank):
        if self.coupling_gain == 0:
            return torch.ones(state.shape[:2], device=self.device)
        phases = self._phases(state, bank)
        correction = self.coupling_gain * (phases[:, 1] - phases[:, 0] - torch.pi).sin()
        return torch.stack([1 + correction, 1 - correction], dim=1)

    @torch.no_grad()
    def _refresh(self, ids):
        self.position[ids] = self.state[ids].flatten(1) * self.signs
        speed = self.speed[ids]
        velocity = self.model(self.state[ids].reshape(-1, 2), speed.repeat_interleave(2)).reshape(-1, 2, 2)
        velocity *= self._coupling_rates(self.state[ids], self._bank_at(speed))[:, :, None]
        self.velocity[ids] = velocity.flatten(1) * self.signs

    @torch.no_grad()
    def reset(self, ids, command_speed):
        self.speed[ids] = command_speed
        bank = self._bank_at(self.speed[ids])
        phase = torch.randint(bank.shape[1], (len(ids),), device=self.device)
        index = torch.stack([phase, (phase + bank.shape[1] // 2) % bank.shape[1]], 1)
        self.state[ids] = bank[torch.arange(len(ids), device=self.device)[:, None], index]
        self._refresh(ids)

    @torch.no_grad()
    def step(self, command_speed, dt):
        self.speed.copy_(command_speed)
        parameters = self.model.parameters_at(self.speed.repeat_interleave(2))
        # Hold the phase feedback for one control interval, like the commanded speed.
        rates = self._coupling_rates(self.state, self._bank_at(self.speed)).reshape(-1, 1)
        state = self.state.reshape(-1, 2)
        h = dt / self.substeps
        def field(q):
            return self.model.field(q, parameters) * rates
        for _ in range(self.substeps):
            a = field(state); b = field(state + h * a / 2)
            c = field(state + h * b / 2); d = field(state + h * c)
            state = state + h * (a + 2 * b + 2 * c + d) / 6
        self.state.copy_(state.reshape_as(self.state))
        self._refresh(torch.arange(self.num_envs, device=self.device))
        return self.position
