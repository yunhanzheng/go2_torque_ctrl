# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


from .base.legged_robot import LeggedRobot

from .go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO
from .go2.go2_torque.go2_torque import GO2Torque
from .go2.go2_torque_flat_config import GO2TorqueFlatCfg, GO2TorqueFlatCfgPPO
from .go2.go2_torque_gait_config import GO2TorqueGaitCfg, GO2TorqueGaitCfgPPO
from .go2.go2_torque_gait_sac_config import (
    GO2TorqueGaitSACCfg,
    GO2TorqueGaitSACTrainCfg,
    GO2TorqueGaitSACSmokeCfg,
    GO2TorqueGaitSACSmokeTrainCfg,
    GO2TorqueGaitSACPilotCfg,
    GO2TorqueGaitSACPilotTrainCfg,
)
from .go2.go2_torque_fixed_speed_config import (
    GO2TorqueFixedSpeedCfg,
    GO2TorqueFixedSpeedCfgPPO,
)
from .go2.go2_torque_fixed_speed_posture_config import (
    GO2TorqueFixedSpeedPostureCfg,
    GO2TorqueFixedSpeedPostureCfgPPO,
)
from .go2.go2_torque_height_termination import GO2TorqueHeightTermination
from .go2.go2_torque_height_termination_config import (
    GO2TorqueHeightTerminationCfg,
    GO2TorqueHeightTerminationCfgPPO,
)
from .go2.go2_torque_nominal_posture import GO2TorqueNominalPosture
from .go2.go2_torque_nominal_posture_config import (
    GO2TorqueNominalPostureCfg,
    GO2TorqueNominalPostureCfgPPO,
)
from .go2.go2_torque_gait import GO2TorqueGait
from .go2.go2_torque_stand_sac import GO2TorqueStandSAC
from .go2.go2_torque_stand_residual_sac import GO2TorqueStandResidualSAC
from .go2.go2_torque_stand_residual_sac_config import (
    GO2TorqueStandResidualSACCfg,
    GO2TorqueStandResidualSACTrainCfg,
)
from .go2.go2_torque_stand_sac_config import (
    GO2TorqueStandSACCfg,
    GO2TorqueStandSACTrainCfg,
    GO2TorqueStandSACV2Cfg,
    GO2TorqueStandSACV2TrainCfg,
)
from .go2.go2_torque_rough_config import GO2TorqueRoughGaitCfg, GO2TorqueRoughGaitCfgPPO
from .go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .go2.go2_stand_sac_config import (
    GO2StandSACCfg,
    GO2StandSACTrainCfg,
    GO2StandSACSmokeCfg,
    GO2StandSACSmokeTrainCfg,
)
from .go2.go2_walk_sac_config import (
    GO2Walk03SACCfg,
    GO2Walk03SACTrainCfg,
    GO2Walk03SACSmokeCfg,
    GO2Walk03SACSmokeTrainCfg,
)
from .go2.go2_walk_sac import GO2WalkSAC
from .go2.go2_walk_sac_v2_config import (
    GO2Walk03SACV2Cfg,
    GO2Walk03SACV2TrainCfg,
    GO2Walk03SACV2SmokeCfg,
    GO2Walk03SACV2SmokeTrainCfg,
)
from .go2.go2_walk_sac_v3_config import (
    GO2Walk025SACV3Cfg,
    GO2Walk025SACV3TrainCfg,
    GO2Walk025SACV3SmokeCfg,
    GO2Walk025SACV3SmokeTrainCfg,
)
from .go2.go2_walk_gait_sac import GO2WalkGaitSAC
from .go2.go2_walk_sac_v4_config import (
    GO2Walk025SACV4Cfg,
    GO2Walk025SACV4TrainCfg,
    GO2Walk025SACV4SmokeCfg,
    GO2Walk025SACV4SmokeTrainCfg,
)


import os

from legged_gym.utils.task_registry import task_registry


task_registry.register("go2_torque", GO2Torque, GO2TorqueCfg(), GO2TorqueCfgPPO())
task_registry.register("go2_torque_flat", GO2Torque, GO2TorqueFlatCfg(), GO2TorqueFlatCfgPPO())
task_registry.register("go2_torque_gait", GO2TorqueGait, GO2TorqueGaitCfg(), GO2TorqueGaitCfgPPO())
task_registry.register("go2_torque_gait_sac", GO2TorqueGait, GO2TorqueGaitSACCfg(), GO2TorqueGaitSACTrainCfg())
task_registry.register(
    "go2_torque_gait_sac_smoke",
    GO2TorqueGait,
    GO2TorqueGaitSACSmokeCfg(),
    GO2TorqueGaitSACSmokeTrainCfg(),
)
task_registry.register(
    "go2_torque_gait_sac_pilot",
    GO2TorqueGait,
    GO2TorqueGaitSACPilotCfg(),
    GO2TorqueGaitSACPilotTrainCfg(),
)
task_registry.register(
    "go2_torque_flat_fixed_05",
    GO2TorqueGait,
    GO2TorqueFixedSpeedCfg(),
    GO2TorqueFixedSpeedCfgPPO(),
)
task_registry.register(
    "go2_torque_flat_fixed_05_posture",
    GO2TorqueGait,
    GO2TorqueFixedSpeedPostureCfg(),
    GO2TorqueFixedSpeedPostureCfgPPO(),
)
task_registry.register(
    "go2_torque_flat_fixed_05_height_term",
    GO2TorqueHeightTermination,
    GO2TorqueHeightTerminationCfg(),
    GO2TorqueHeightTerminationCfgPPO(),
)
task_registry.register(
    "go2_torque_flat_fixed_05_nominal_posture",
    GO2TorqueNominalPosture,
    GO2TorqueNominalPostureCfg(),
    GO2TorqueNominalPostureCfgPPO(),
)
task_registry.register("go2_torque_rough_gait", GO2TorqueGait, GO2TorqueRoughGaitCfg(), GO2TorqueRoughGaitCfgPPO())
task_registry.register(
    "go2_torque_stand_sac",
    GO2TorqueStandSAC,
    GO2TorqueStandSACCfg(),
    GO2TorqueStandSACTrainCfg(),
)
task_registry.register(
    "go2_torque_stand_sac_v2",
    GO2TorqueStandSAC,
    GO2TorqueStandSACV2Cfg(),
    GO2TorqueStandSACV2TrainCfg(),
)
task_registry.register(
    "go2_torque_stand_residual_sac",
    GO2TorqueStandResidualSAC,
    GO2TorqueStandResidualSACCfg(),
    GO2TorqueStandResidualSACTrainCfg(),
)
task_registry.register("go2_rough", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register(
    "go2_stand_sac",
    LeggedRobot,
    GO2StandSACCfg(),
    GO2StandSACTrainCfg(),
)
task_registry.register(
    "go2_stand_sac_smoke",
    LeggedRobot,
    GO2StandSACSmokeCfg(),
    GO2StandSACSmokeTrainCfg(),
)
task_registry.register(
    "go2_walk_03_sac",
    LeggedRobot,
    GO2Walk03SACCfg(),
    GO2Walk03SACTrainCfg(),
)
task_registry.register(
    "go2_walk_03_sac_smoke",
    LeggedRobot,
    GO2Walk03SACSmokeCfg(),
    GO2Walk03SACSmokeTrainCfg(),
)
task_registry.register(
    "go2_walk_03_sac_v2",
    GO2WalkSAC,
    GO2Walk03SACV2Cfg(),
    GO2Walk03SACV2TrainCfg(),
)
task_registry.register(
    "go2_walk_03_sac_v2_smoke",
    GO2WalkSAC,
    GO2Walk03SACV2SmokeCfg(),
    GO2Walk03SACV2SmokeTrainCfg(),
)
task_registry.register(
    "go2_walk_025_sac_v3",
    GO2WalkSAC,
    GO2Walk025SACV3Cfg(),
    GO2Walk025SACV3TrainCfg(),
)
task_registry.register(
    "go2_walk_025_sac_v3_smoke",
    GO2WalkSAC,
    GO2Walk025SACV3SmokeCfg(),
    GO2Walk025SACV3SmokeTrainCfg(),
)
task_registry.register(
    "go2_walk_025_sac_v4",
    GO2WalkGaitSAC,
    GO2Walk025SACV4Cfg(),
    GO2Walk025SACV4TrainCfg(),
)
task_registry.register(
    "go2_walk_025_sac_v4_smoke",
    GO2WalkGaitSAC,
    GO2Walk025SACV4SmokeCfg(),
    GO2Walk025SACV4SmokeTrainCfg(),
)
