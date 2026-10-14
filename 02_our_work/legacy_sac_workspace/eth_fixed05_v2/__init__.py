"""Fixed-speed upright Go2 task for the ETH official SAC implementation."""

import gymnasium as gym
import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_sac_cfg import (
    UnitreeGo2FlatSACRunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
    UnitreeGo2FlatEnvCfg_PLAY,
)


def delayed_low_base(env, minimum_height: float, grace_steps: int) -> torch.Tensor:
    """End persistent crawling, while leaving initial exploration time to recover."""
    robot = env.scene["robot"]
    return (env.episode_length_buf > grace_steps) & (robot.data.root_pos_w[:, 2] < minimum_height)


def gait_clock(env, frequency: float = 1.5) -> torch.Tensor:
    """SATA Stage-3 sinusoidal gait phase, expressed at the policy rate."""
    phase = 2.0 * torch.pi * frequency * env.episode_length_buf * env.step_dt
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def sata_policy_observation(env, frequency: float = 1.5) -> torch.Tensor:
    """Return the SATA Stage-3 actor observation in its exact 62-D order.

    Isaac Lab's Go2 model has no SATA muscle-fatigue state.  The corresponding
    12 channels are therefore explicit zeros instead of silently changing the
    actor input layout.
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    # Isaac Lab stores the Go2 joints grouped by joint type, whereas SATA uses
    # FL/FR/RL/RR with hip/thigh/calf contiguous for each leg.
    sata_joint_names = (
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    )
    joint_ids, _ = robot.find_joints(list(sata_joint_names), preserve_order=True)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    joint_vel = robot.data.joint_vel[:, joint_ids]
    applied_torque = robot.data.applied_torque[:, joint_ids]
    fatigue_placeholder = torch.zeros_like(joint_pos)
    return torch.cat(
        (
            robot.data.root_lin_vel_b * 2.0,
            robot.data.root_ang_vel_b * 0.25,
            robot.data.projected_gravity_b,
            (joint_pos - default_joint_pos),
            joint_vel * 0.05,
            command * command.new_tensor((2.0, 2.0, 0.25)),
            applied_torque,
            fatigue_placeholder,
            gait_clock(env, frequency),
        ),
        dim=-1,
    )


def _apply_sata_observations(cfg):
    # Use one composite term so the manager cannot reorder SATA's channels.
    cfg.observations.policy.base_lin_vel = ObsTerm(
        func=sata_policy_observation, params={"frequency": 1.5}
    )
    cfg.observations.policy.base_ang_vel = None
    cfg.observations.policy.projected_gravity = None
    cfg.observations.policy.velocity_commands = None
    cfg.observations.policy.joint_pos = None
    cfg.observations.policy.joint_vel = None
    cfg.observations.policy.actions = None
    cfg.observations.policy.enable_corruption = False


def _apply_fixed_upright_settings(cfg):
    # One deliberately simple learning target: straight walking at 0.5 m/s.
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (20.0, 20.0)
    command.ranges.lin_vel_x = (0.5, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

    # Prevent the low-body sliding solution observed in v1.
    cfg.rewards.base_height_l2 = RewTerm(
        func=base_mdp.base_height_l2,
        weight=-12.0,
        params={"target_height": 0.36},
    )
    cfg.rewards.undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )
    cfg.rewards.feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    # A larger air-time signal than the stock Go2 value makes stepping preferable
    # to keeping all four feet planted and shuffling.
    cfg.rewards.feet_air_time.weight = 0.5
    cfg.rewards.feet_air_time.params["threshold"] = 0.35

    cfg.terminations.low_base = DoneTerm(
        func=delayed_low_base,
        params={"minimum_height": 0.20, "grace_steps": 100},
    )


def _apply_fixed_05_command_only(cfg):
    """Set the paper comparison command without changing the ETH task MDP."""
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (20.0, 20.0)
    command.ranges.lin_vel_x = (0.5, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)


@configclass
class Go2Fixed05UprightEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_upright_settings(self)


@configclass
class Go2Fixed05UprightEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_upright_settings(self)


@configclass
class Go2Fixed05UprightSACRunnerCfg(UnitreeGo2FlatSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_go2_fixed_05_upright_v2"


@configclass
class Go2Fixed05Upright200HzEnvCfg(UnitreeGo2FlatEnvCfg):
    """Frequency-only ablation of v2: policy/control loop at 200 Hz."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_upright_settings(self)
        _apply_sata_observations(self)
        self.decimation = 1
        self.sim.render_interval = self.decimation
        # Keep the low-height recovery grace period at two physical seconds.
        self.terminations.low_base.params["grace_steps"] = 400


@configclass
class Go2Fixed05Upright200HzEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_upright_settings(self)
        _apply_sata_observations(self)
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.terminations.low_base.params["grace_steps"] = 400


@configclass
class Go2Fixed05Upright200HzSACRunnerCfg(UnitreeGo2FlatSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_go2_fixed_05_upright_200hz_v3"


@configclass
class Go2Fixed05UprightPolicyFreq200EnvCfg(UnitreeGo2FlatEnvCfg):
    """Teacher-requested ablation: stock control rate and SATA observations."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_upright_settings(self)
        _apply_sata_observations(self)


@configclass
class Go2Fixed05UprightPolicyFreq200EnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_upright_settings(self)
        _apply_sata_observations(self)


@configclass
class Go2Fixed05UprightPolicyFreq200SACRunnerCfg(UnitreeGo2FlatSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_sac_go2_fixed05_policyfreq200_sataobs_v4"
        self.algorithm.policy_frequency = 200


@configclass
class Go2EthOfficialFixed05PolicyFreq200SATAObsEnvCfg(UnitreeGo2FlatEnvCfg):
    """ETH official Flat MDP plus only the two teacher-requested changes."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_05_command_only(self)
        _apply_sata_observations(self)


@configclass
class Go2EthOfficialFixed05PolicyFreq200SATAObsEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_05_command_only(self)
        _apply_sata_observations(self)


@configclass
class Go2EthOfficialFixed05PolicyFreq200SATAObsSACRunnerCfg(UnitreeGo2FlatSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_official_fixed05_policyfreq200_sataobs_v5"
        self.algorithm.policy_frequency = 200


@configclass
class Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg(UnitreeGo2FlatEnvCfg):
    """ETH official Flat MDP with 200-Hz policy execution and SATA observations."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_05_command_only(self)
        _apply_sata_observations(self)
        self.decimation = 1
        self.sim.render_interval = self.decimation


@configclass
class Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_05_command_only(self)
        _apply_sata_observations(self)
        self.decimation = 1
        self.sim.render_interval = self.decimation


@configclass
class Go2EthOfficialFixed05Policy200HzSATAObsSACRunnerCfg(UnitreeGo2FlatSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_official_fixed05_policy200hz_sataobs_v6"


@configclass
class Go2EthOfficialFixed05Policy200HzTimeAlignedSATAObsSACRunnerCfg(
    UnitreeGo2FlatSACRunnerCfg
):
    """ETH SAC at 200 Hz with its original 50-Hz physical time horizons."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "eth_official_fixed05_policy200hz_timealigned_sataobs_v7"
        # Preserve gamma=0.96 per 20 ms after changing the policy step to 5 ms.
        self.algorithm.gamma = 0.96 ** 0.25
        # Preserve the official 5-step (100 ms) n-step return horizon.
        self.algorithm.n_steps = 20
        # Keep the official memory footprint per collection. Four times as many
        # iterations preserve the total physical experience at the 4x rate.
        self.num_steps_per_env = 24
        self.max_iterations = 2400
        # This is actor-update frequency, not policy execution frequency.
        self.algorithm.policy_frequency = 1


gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Fixed05-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2Fixed05UprightEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2Fixed05UprightSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Fixed05-PolicyFreq200-SATAObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2Fixed05UprightPolicyFreq200EnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2Fixed05UprightPolicyFreq200SACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Fixed05-PolicyFreq200-SATAObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2Fixed05UprightPolicyFreq200EnvCfg_PLAY",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2Fixed05UprightPolicyFreq200SACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHOfficial-Fixed05-PolicyFreq200-SATAObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05PolicyFreq200SATAObsEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05PolicyFreq200SATAObsSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHOfficial-Fixed05-PolicyFreq200-SATAObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05PolicyFreq200SATAObsEnvCfg_PLAY",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05PolicyFreq200SATAObsSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHOfficial-Fixed05-Policy200Hz-SATAObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05Policy200HzSATAObsSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHOfficial-Fixed05-Policy200Hz-SATAObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg_PLAY",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05Policy200HzSATAObsSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-ETHOfficial-Fixed05-Policy200Hz-TimeAligned-SATAObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2EthOfficialFixed05Policy200HzSATAObsEnvCfg",
        "rsl_rl_sac_cfg_entry_point": (
            f"{__name__}:Go2EthOfficialFixed05Policy200HzTimeAlignedSATAObsSACRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Fixed05-Upright-200Hz-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2Fixed05Upright200HzEnvCfg",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2Fixed05Upright200HzSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Fixed05-Upright-200Hz-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2Fixed05Upright200HzEnvCfg_PLAY",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2Fixed05Upright200HzSACRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Fixed05-Upright-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:Go2Fixed05UprightEnvCfg_PLAY",
        "rsl_rl_sac_cfg_entry_point": f"{__name__}:Go2Fixed05UprightSACRunnerCfg",
    },
)
