"""Isaac Lab ETH-SAC task with SATA Stage-3 reward terms."""

import gymnasium as gym
import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from eth_fixed05_v2 import (
    Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg,
    Go2EthOfficialFixed05Policy200HzTimeAlignedSATAObsSACRunnerCfg,
)


def sata_forward(env, command_name: str, sigma: float = 0.25):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    return torch.exp(-torch.abs(robot.data.root_lin_vel_b[:, 0] - command[:, 0]) / sigma)


def sata_lateral(env, command_name: str, sigma: float = 0.25):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    return torch.exp(-torch.abs(robot.data.root_lin_vel_b[:, 1] - command[:, 1]) / sigma)


def sata_yaw(env, command_name: str, sigma: float = 0.25):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    return torch.exp(-torch.abs(robot.data.root_ang_vel_b[:, 2] - command[:, 2]) / sigma)


def sata_head_height(env, target_height: float = 0.3):
    robot = env.scene["robot"]
    height = robot.data.root_pos_w[:, 2].clip(max=target_height)
    head_up = -robot.data.projected_gravity_b[:, 0].clip(min=0.0)
    return 2.0 * height + head_up


def sata_roll(env):
    return torch.abs(env.scene["robot"].data.projected_gravity_b[:, 1])


def _desired_contacts(env, frequency: float):
    phase = 2.0 * torch.pi * frequency * env.episode_length_buf * env.step_dt
    first_diagonal = torch.sin(phase) >= 0.0
    desired = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
    desired[:, 0] = first_diagonal
    desired[:, 3] = first_diagonal
    desired[:, 1] = ~first_diagonal
    desired[:, 2] = ~first_diagonal
    return desired


def sata_gait_contact(env, sensor_cfg: SceneEntityCfg, command_name: str, frequency: float = 1.5):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    actual = forces > 1.0
    desired = _desired_contacts(env, frequency)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return (actual == desired).float().mean(dim=1) * moving


def sata_swing_clearance(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    frequency: float = 1.5,
    swing_height: float = 0.065,
    sigma: float = 0.025,
):
    robot = env.scene[asset_cfg.name]
    desired_swing = ~_desired_contacts(env, frequency)
    height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    score = torch.exp(-torch.square((height - swing_height) / sigma))
    moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return (score * desired_swing.float()).sum(dim=1) / 2.0 * moving


def sata_feet_slip(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    robot = env.scene[asset_cfg.name]
    speed_sq = torch.sum(torch.square(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1)
    return torch.sum(speed_sq * contact.float(), dim=1)


def _apply_sata_stage3_rewards(cfg):
    feet_sensor = SceneEntityCfg("contact_forces", body_names=".*_foot")
    feet_asset = SceneEntityCfg("robot", body_names=".*_foot")

    cfg.rewards.track_lin_vel_xy_exp = RewTerm(
        func=sata_forward, weight=10.0, params={"command_name": "base_velocity", "sigma": 0.25}
    )
    cfg.rewards.track_ang_vel_z_exp = RewTerm(
        func=sata_yaw, weight=5.0, params={"command_name": "base_velocity", "sigma": 0.25}
    )
    cfg.rewards.lin_vel_z_l2.weight = -5.0
    cfg.rewards.ang_vel_xy_l2 = None
    cfg.rewards.dof_torques_l2 = None
    cfg.rewards.dof_acc_l2.weight = -1.0e-6
    cfg.rewards.action_rate_l2 = None
    cfg.rewards.feet_air_time = None
    cfg.rewards.flat_orientation_l2 = None
    cfg.rewards.dof_pos_limits.weight = -5.0
    cfg.rewards.undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"]
            ),
            "threshold": 1.0,
        },
    )
    cfg.rewards.head_height = RewTerm(func=sata_head_height, weight=5.0, params={"target_height": 0.3})
    cfg.rewards.moving_y = RewTerm(
        func=sata_lateral, weight=5.0, params={"command_name": "base_velocity", "sigma": 0.25}
    )
    cfg.rewards.roll = RewTerm(func=sata_roll, weight=-5.0)
    cfg.rewards.gait_contact = RewTerm(
        func=sata_gait_contact,
        weight=3.0,
        params={"sensor_cfg": feet_sensor, "command_name": "base_velocity", "frequency": 1.5},
    )
    cfg.rewards.swing_clearance = RewTerm(
        func=sata_swing_clearance,
        weight=1.5,
        params={
            "sensor_cfg": feet_sensor,
            "asset_cfg": feet_asset,
            "command_name": "base_velocity",
            "frequency": 1.5,
            "swing_height": 0.065,
            "sigma": 0.025,
        },
    )
    cfg.rewards.feet_slip = RewTerm(
        func=sata_feet_slip,
        weight=-0.5,
        params={"sensor_cfg": feet_sensor, "asset_cfg": feet_asset},
    )


@configclass
class Go2EthSACSATARewardEnvCfg(Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_sata_stage3_rewards(self)


@configclass
class Go2EthSACSATARewardRunnerCfg(Go2EthOfficialFixed05Policy200HzTimeAlignedSATAObsSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_isaaclab_sata_reward_v8"
        self.max_iterations = 1000


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHSAC-SATAReward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthSACSATARewardRunnerCfg",
    },
)
