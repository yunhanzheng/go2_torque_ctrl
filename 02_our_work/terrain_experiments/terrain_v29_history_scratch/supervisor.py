"""Adapt the V28 guarded Pilot workflow to the V29 history policy."""
from pathlib import Path


path = Path("/home/ubuntu22/terrain_v28_scratch_pilot/supervisor.py")
source = path.read_text()
replacements = {
    "V28": "V29",
    "v28": "v29",
    "terrain_v28_scratch_pilot": "terrain_v29_history_scratch",
    "terrain_v28_scratch_multiterrain": "terrain_v29_history_scratch_multiterrain",
    "Isaac-Go2-Terrain-V28-ScratchMulti-v0": "Isaac-Go2-Terrain-V29-HistoryScratch-v0",
    '"--max_iterations", "300"': '"--max_iterations", "300"',
    '"agent.algorithm.replay_buffer_size=8192"': '"agent.algorithm.replay_buffer_size=4096"',
}
for old, new in replacements.items():
    source = source.replace(old, new)
source = source.replace("terrain_v29_scratch_multiterrain", "terrain_v29_history_scratch_multiterrain")
source = source.replace("Isaac-Go2-Terrain-V29-ScratchMulti-v0", "Isaac-Go2-Terrain-V29-HistoryScratch-v0")
source = source.replace(
    "随机初始化；官方高度扫描观测；",
    "随机初始化；当前高度扫描＋过去160 ms五帧历史；",
)
source = source.replace(
    "Policy使用官方高度扫描观测。",
    "Policy使用当前官方高度扫描观测，并通过Encoder使用过去160 ms历史。",
)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": __file__})
