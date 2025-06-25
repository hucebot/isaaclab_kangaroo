# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Fabio Amadio (fabioamadio93@gmail.com).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .kangaroo_locomotion_rough_env_cfg import KangarooRoughEnvCfg


@configclass
class KangarooTerminationsCfg:
    """Kangaroo Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class KangarooFlatEnvCfg(KangarooRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations = KangarooTerminationsCfg()
