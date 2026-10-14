"""Simplified flat-ground SATA task used for baseline reproduction."""

from legged_gym.envs.go2.go2_torque.go2_torque_config import (
    GO2TorqueCfg,
    GO2TorqueCfgPPO,
)


class GO2TorqueFlatCfg(GO2TorqueCfg):
    class env(GO2TorqueCfg.env):
        num_envs = 4096
        episode_length_s = 10

    class init_state(GO2TorqueCfg.init_state):
        # Start from a normal standing height for the first flat-ground baseline.
        pos = [0.0, 0.0, 0.30]

    class terrain(GO2TorqueCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(GO2TorqueCfg.commands):
        heading_command = False
        resampling_time = 5

        class ranges:
            # First learn symmetric standing and low-speed forward locomotion.
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class noise(GO2TorqueCfg.noise):
        add_noise = False

    class rewards(GO2TorqueCfg.rewards):
        class scales(GO2TorqueCfg.rewards.scales):
            # Prevent the knee/calf-walking shortcut observed in stage 1.
            collision = -5.0
            # Encourage alternating contacts made by the actual feet.
            feet_air_time = 1.0

    class domain_rand(GO2TorqueCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False
        loss_action_obs = False


class GO2TorqueFlatCfgPPO(GO2TorqueCfgPPO):
    class runner(GO2TorqueCfgPPO.runner):
        experiment_name = "SATA_flat_foot_contact"
        run_name = "stage2_collision_airtime"
        max_iterations = 3000
