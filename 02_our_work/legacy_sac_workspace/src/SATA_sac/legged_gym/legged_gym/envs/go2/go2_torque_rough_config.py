"""Stage-4 terrain curriculum initialized from the Stage-3 flat gait policy."""

from legged_gym.envs.go2.go2_torque_gait_config import (
    GO2TorqueGaitCfg,
    GO2TorqueGaitCfgPPO,
)


class GO2TorqueRoughGaitCfg(GO2TorqueGaitCfg):
    class terrain(GO2TorqueGaitCfg.terrain):
        mesh_type = "trimesh"
        # Keep the Stage-3 observation size so its actor weights can be reused.
        # The first rough policy is proprioceptive; terrain heights can be added
        # in a later teacher/student experiment.
        measure_heights = False
        curriculum = True
        max_init_terrain_level = 0
        num_rows = 6
        num_cols = 8
        terrain_length = 8.0
        terrain_width = 8.0
        # smooth slopes, then rough slopes; no stairs/gaps/pits in Stage 4A.
        terrain_proportions = [0.15, 0.85, 0.0, 0.0, 0.0]


class GO2TorqueRoughGaitCfgPPO(GO2TorqueGaitCfgPPO):
    class runner(GO2TorqueGaitCfgPPO.runner):
        # Keep this root so --load_run can resolve the Stage-3 checkpoint.
        experiment_name = "SATA_flat_gait_coordination"
        run_name = "stage4a_rough_curriculum"
        max_iterations = 2000
