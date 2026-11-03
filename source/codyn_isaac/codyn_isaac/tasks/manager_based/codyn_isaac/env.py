"""Advance reference once per control step; reset only terminated environments."""

from isaaclab.envs import ManagerBasedRLEnv
from .mdp.observations import get_reference


class G1CoDynEnv(ManagerBasedRLEnv):
    def step(self, action):
        get_reference(self).step(self.command_manager.get_command("base_velocity")[:, 0], self.step_dt)
        return super().step(action)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "gait_reference"):
            command = self.command_manager.get_command("base_velocity")[env_ids, 0]
            self.gait_reference.reset(env_ids, command)
