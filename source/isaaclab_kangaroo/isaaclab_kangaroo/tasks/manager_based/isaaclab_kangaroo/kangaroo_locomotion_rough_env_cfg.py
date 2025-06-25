# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Fabio Amadio (fabioamadio93@gmail.com).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import mdp

##
# Kangaroo configs
##
from isaaclab_kangaroo import KANGAROO_MINIMAL_CFG  # isort: skip


@configclass
class KangarooRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )

    # Penalize uneven step times between the two feet
    different_step_times = RewTerm(
        func=mdp.different_step_times,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
        },
    )


@configclass
class KangarooActionsCfg:
    """Kangaroo Action specifications for the MDP."""

    joint_pos = mdp.CustomJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*_motor"],
        scale=0.1,
        use_tanh=True,
        clamp_offset=[0, 0, 0, 0, 0, 0, 0, 0, -0.54, -0.54, 0, 0],
    )


@configclass
class KangarooObservationsCfg:
    """Kangaroo Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_motor",  # joint_idx: 4
                        "leg_right_1_motor",  # joint_idx: 11
                        "leg_left_2_motor",  # joint_idx: 18
                        "leg_left_3_motor",  # joint_idx: 19
                        "leg_right_2_motor",  # joint_idx: 36
                        "leg_right_3_motor",  # joint_idx: 37
                        "leg_left_4_motor",  # joint_idx: 38
                        "leg_left_5_motor",  # joint_idx: 40
                        "leg_left_length_motor",  # joint_idx: 43
                        "leg_right_length_motor",  # joint_idx: 45
                        "leg_right_4_motor",  # joint_idx: 46
                        "leg_right_5_motor",  # joint_idx: 48
                    ],
                )
            },
            noise=Unoise(n_min=-0.0025, n_max=0.0025),
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_motor",  # joint_idx: 4
                        "leg_right_1_motor",  # joint_idx: 11
                        "leg_left_2_motor",  # joint_idx: 18
                        "leg_left_3_motor",  # joint_idx: 19
                        "leg_right_2_motor",  # joint_idx: 36
                        "leg_right_3_motor",  # joint_idx: 37
                        "leg_left_4_motor",  # joint_idx: 38
                        "leg_left_5_motor",  # joint_idx: 40
                        "leg_left_length_motor",  # joint_idx: 43
                        "leg_right_length_motor",  # joint_idx: 45
                        "leg_right_4_motor",  # joint_idx: 46
                        "leg_right_5_motor",  # joint_idx: 48
                    ],
                )
            },
            noise=Unoise(n_min=-0.025, n_max=0.025),
        )
        measured_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_joint",  # joint_idx: 1
                        "leg_right_1_joint",  # joint_idx: 2
                        "leg_left_2_joint",  # joint_idx: 7
                        "leg_right_2_joint",  # joint_idx: 8
                        "leg_left_3_joint",  # joint_idx: 14
                        "leg_right_3_joint",  # joint_idx: 15
                        "left_ankle_4_pendulum_joint",  # joint_idx: 21
                        "left_ankle_5_pendulum_joint",  # joint_idx: 23
                        "right_ankle_4_pendulum_joint",  # joint_idx: 30
                        "right_ankle_5_pendulum_joint",  # joint_idx: 32
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        measured_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_joint",  # joint_idx: 1
                        "leg_right_1_joint",  # joint_idx: 2
                        "leg_left_2_joint",  # joint_idx: 7
                        "leg_right_2_joint",  # joint_idx: 8
                        "leg_left_3_joint",  # joint_idx: 14
                        "leg_right_3_joint",  # joint_idx: 15
                        "left_ankle_4_pendulum_joint",  # joint_idx: 21
                        "left_ankle_5_pendulum_joint",  # joint_idx: 23
                        "right_ankle_4_pendulum_joint",  # joint_idx: 30
                        "right_ankle_5_pendulum_joint",  # joint_idx: 32
                    ],
                )
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class KangarooEventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.4, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0, 0),
                "y": (-0, 0),
                "z": (-0, 0),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (-0, 0),
            },
        },
    )


@configclass
class KangarooViewerCfg(ViewerCfg):
    # HD: 1280 x 720; Full HD: 1920 x 1080; 4K: 3840 x 2160
    resolution: tuple[int, int] = (1920, 1080)
    # Default: (7.5, 7.5, 7.5)
    eye: tuple[float, float, float] = (10, 10, 10)


@configclass
class KangarooRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KangarooRewards = KangarooRewards()
    actions: KangarooActionsCfg = KangarooActionsCfg()
    observations: KangarooObservationsCfg = KangarooObservationsCfg()
    events: KangarooEventCfg = KangarooEventCfg()
    viewer: KangarooViewerCfg = KangarooViewerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.num_envs = 2048
        self.scene.robot = KANGAROO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso"
        self.scene.height_scanner.offset.pos = (0.0, 0.0, 0.0)
        self.scene.height_scanner.pattern_cfg.size = [1.2, 1.2]
        self.scene.height_scanner.debug_vis = True

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_pos_limits = None
        self.rewards.dof_acc_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.undesired_contacts = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso"
