import json
import sys

import numpy as np

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg

from terrain_v26_official3.config import terrain_catalog


result = []
for name, sub_terrain in terrain_catalog().items():
    for difficulty in (0.25, 0.75):
        sub_terrain.proportion = 1.0
        generator = TerrainGenerator(
            TerrainGeneratorCfg(
                seed=7,
                size=(8.0, 8.0),
                num_rows=1,
                num_cols=1,
                difficulty_range=(difficulty, difficulty),
                sub_terrains={name: sub_terrain},
                use_cache=False,
            ),
            device="cpu",
        )
        assert np.isfinite(generator.terrain_mesh.vertices).all(), name
        assert np.isfinite(generator.terrain_origins).all(), name
        result.append({"terrain": name, "difficulty": difficulty, "vertices": len(generator.terrain_mesh.vertices)})

with open(sys.argv[1], "w") as stream:
    json.dump(result, stream, indent=2)
print("[PREFLIGHT_OK]", len(result), "terrain/difficulty combinations", flush=True)
simulation_app.close()
