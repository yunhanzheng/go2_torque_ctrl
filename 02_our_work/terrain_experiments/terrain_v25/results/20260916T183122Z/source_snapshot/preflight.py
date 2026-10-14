from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import json
import sys
from pathlib import Path
import numpy as np
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from terrain_v25.config import terrain_catalog

result=[]
for name, sub in terrain_catalog().items():
    assert 'Stairs' not in type(sub).__name__
    for d in (.25,.75):
        gen=TerrainGenerator(TerrainGeneratorCfg(seed=7,size=(10.,10.),num_rows=1,num_cols=1,
            horizontal_scale=.1,vertical_scale=.005,slope_threshold=.75,curriculum=False,
            difficulty_range=(d,d),sub_terrains={name:sub},use_cache=False),device='cpu')
        assert np.isfinite(gen.terrain_mesh.vertices).all(),name
        assert np.isfinite(gen.terrain_origins).all(),name
        result.append({'terrain':name,'difficulty':d,'vertices':len(gen.terrain_mesh.vertices),
                       'origin':gen.terrain_origins.tolist(),'config':sub.to_dict()})
Path(sys.argv[1]).write_text(json.dumps(result,default=str,indent=2))
print('[PREFLIGHT_OK]',len(result),'terrain/difficulty combinations',flush=True)
app.close()
