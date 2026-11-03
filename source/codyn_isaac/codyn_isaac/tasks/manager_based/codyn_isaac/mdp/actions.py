"""Keep the 29-joint action order, replacing four default offsets with the gait reference."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from .observations import oscillator_position
from .reference import JOINT_NAMES


class OscillatorResidualJointPositionAction(JointPositionAction):
    """Hip/knee targets = reference + scaled action; all other targets stay unchanged.

    Reference positions already include the G1 hip signs and reference soft-limit
    clipping. Use the observation's cached control-step reference and hold targets
    across physics substeps. Do not add the default pose to the four references.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        missing = set(JOINT_NAMES) - set(self._joint_names)
        if missing:
            raise ValueError(f"Residual action is missing reference joints: {sorted(missing)}")
        self._reference_action_ids = [self._joint_names.index(name) for name in JOINT_NAMES]

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        ids = self._reference_action_ids
        scale = self._scale[:, ids] if isinstance(self._scale, torch.Tensor) else self._scale
        targets = oscillator_position(self._env) + self._raw_actions[:, ids] * scale
        if self.cfg.clip is not None:
            targets = targets.clamp(self._clip[:, ids, 0], self._clip[:, ids, 1])
        self._processed_actions[:, ids] = targets


@configclass
class OscillatorResidualJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = OscillatorResidualJointPositionAction
