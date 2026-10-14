"""Run the existing fixed-stair first-episode protocol with history-aware tasks."""
from pathlib import Path

base = Path('/home/ubuntu22/eval_eth_sata_v17_stairs.py')
source = base.read_text()
old = 'import eth_sata_reward_v17_stairs'
assert source.count(old) == 1
source = source.replace(old, old + '\nimport stair_v24.config')
old = '    # create runner from rsl-rl'
# The original evaluation has no learning; avoid allocating a training-sized replay.
needle = '    # load previously trained model'
assert source.count(needle) == 1
source = source.replace(needle, '    agent_cfg.algorithm.replay_buffer_size = 4096\n' + needle)
# Zero-height video uses a true plane mesh while preserving the height scanner.
needle = '    gen.sub_terrains = {key: stair}'
assert source.count(needle) == 1
source = source.replace(needle, needle + "\n    if height == 0.:\n        from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg\n        gen.sub_terrains = {'plane': MeshPlaneTerrainCfg(proportion=1.0)}")
# Preserve the 200 Hz controller and 12 s duration, render/capture at 50 Hz.
needle = '    # create isaac environment\n    # set viewer camera before creating isaac environment'
assert source.count(needle) == 1
source = source.replace(needle, '    if args_cli.video:\n        env_cfg.sim.render_interval = 4\n' + needle)
needle = '        env = gym.wrappers.RecordVideo(env, **video_kwargs)'
assert source.count(needle) == 1
source = source.replace(needle, '''        class Record50Hz(gym.wrappers.RecordVideo):
            def _capture_frame(self):
                if self.step_id % 4 == 3:
                    super()._capture_frame()
        env = Record50Hz(env, fps=50, **video_kwargs)''')
source = source.replace('env_cfg.viewer.eye = (2.2, -2.2, 1.35)', 'env_cfg.viewer.eye = (0.7, -2.7, 0.9)')
exec(compile(source, str(base), 'exec'), {'__name__': '__main__', '__file__': str(base)})
