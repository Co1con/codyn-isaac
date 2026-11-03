"""Frozen inference class exported verbatim from code/codyn-model/model.py."""
import torch
from torch import nn

class SpeedOscillator(nn.Module):
    """二维 PHS 风格骨架＋正半定整形矩阵＋补偿；速度网络调制隐藏层。"""

    def __init__(self, geometry, hidden=48, bounded_compensation=False):
        super().__init__()
        self.register_buffer("geometry_base", torch.tensor(geometry["coefficients"], dtype=torch.float32))
        self.register_buffer("geometry_scale", torch.tensor([0.10, 0.10, 0.25, 0.25, 0.25, 0.25]))
        self.midpoint, self.half_range, self.sign = geometry["midpoint"], geometry["half_range"], geometry["sign"]
        self.bounded_compensation = bounded_compensation
        self.speed_net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 6 + 2 * hidden))
        self.state_net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.shaping = nn.Linear(hidden, 5)
        nn.init.zeros_(self.speed_net[-1].weight); nn.init.zeros_(self.speed_net[-1].bias)
        nn.init.normal_(self.shaping.weight, std=0.01); nn.init.zeros_(self.shaping.bias)

    def parameters_at(self, speed):
        normalized = (speed.reshape(-1, 1) - self.midpoint) / self.half_range
        condition = self.speed_net(normalized)
        base = torch.cat([torch.ones_like(normalized), normalized], dim=1) @ self.geometry_base
        parameters = base + self.geometry_scale * torch.tanh(condition[:, :6])
        center, axes = parameters[:, :2], parameters[:, 2:4].exp()
        theta, frequency = parameters[:, 4], parameters[:, 5].exp()
        c, s = theta.cos(), theta.sin()
        basis = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=1)
        return center, axes, basis, frequency, condition[:, 6:].chunk(2, dim=1)

    def field(self, q, parameters):
        center, axes, basis, frequency, (gain, bias) = parameters
        offset = q - center
        rotated = torch.bmm(offset[:, None], basis).squeeze(1)
        normalized = rotated / axes
        gradient = torch.bmm((rotated / axes.square())[:, None], basis.transpose(1, 2)).squeeze(1)
        h = self.state_net(normalized)
        output = self.shaping(h * (1 + 0.25 * gain.tanh()) + 0.25 * bias.tanh())
        a, b, c = (0.1 * output[:, :3].tanh()).unbind(-1)
        zero = torch.zeros_like(a)
        lower = torch.stack([torch.stack([a, zero], -1), torch.stack([b, c], -1)], 1)
        damping = torch.bmm(lower @ lower.transpose(1, 2), gradient[:, :, None]).squeeze(-1)
        circulation = self.sign * 2 * torch.pi * frequency[:, None] * axes.prod(-1, keepdim=True) * torch.stack([-gradient[:, 1], gradient[:, 0]], -1)
        radial = 5 * (normalized.square().sum(-1, keepdim=True) - 1) * offset
        compensation = output[:, 3:].tanh() if self.bounded_compensation else output[:, 3:]
        return circulation - radial - damping + 2 * compensation

    def forward(self, q, speed):
        return self.field(q, self.parameters_at(speed))

    def trajectory(self, initial, speed, duration, steps=100, substeps=2):
        """固定步长 RK4，batch 每条轨迹可具有不同速度和积分时长；保留完整梯度。"""
        parameters = self.parameters_at(speed)
        dt = duration.reshape(-1, 1) / (steps * substeps)
        q, states = initial, [initial]
        for _ in range(steps):
            for _ in range(substeps):
                k1 = self.field(q, parameters)
                k2 = self.field(q + dt * k1 / 2, parameters)
                k3 = self.field(q + dt * k2 / 2, parameters)
                k4 = self.field(q + dt * k3, parameters)
                q = q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            states.append(q)
        return torch.stack(states, dim=1)
