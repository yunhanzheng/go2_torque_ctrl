"""Adapt the guarded workflow to the V30 history curriculum policy."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v28_scratch_pilot/supervisor.py")
source = path.read_text()
replacements = {
    "V28": "V30",
    "v28": "v30",
    "terrain_v28_scratch_pilot": "terrain_v30_history_curriculum",
    "terrain_v28_scratch_multiterrain": "terrain_v30_history_curriculum_multiterrain",
    "Isaac-Go2-Terrain-V28-ScratchMulti-v0": "Isaac-Go2-Terrain-V30-HistoryCurriculum-v0",
    '"--max_iterations", "300"': '"--max_iterations", "400"',
    "Pilot_300": "Pilot_400",
    "model_299.pt": "model_399.pt",
    '"agent.algorithm.replay_buffer_size=8192"': '"agent.algorithm.replay_buffer_size=4096"',
}
for old, new in replacements.items():
    source = source.replace(old, new)
# The broad V28->V30 label replacement runs before the full task-id mapping.
source = source.replace(
    "Isaac-Go2-Terrain-V30-ScratchMulti-v0",
    "Isaac-Go2-Terrain-V30-HistoryCurriculum-v0",
)
source = source.replace(
    "随机初始化；官方高度扫描观测；",
    "随机初始化；0.5 m/s固定前进；最低难度起步；当前高度扫描＋过去160 ms五帧历史；",
)
source = source.replace(
    "Policy使用官方高度扫描观测。",
    "Policy使用当前官方高度扫描观测及过去160 ms历史；四类地形从最低难度开始并由课程升阶。",
)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": __file__})
