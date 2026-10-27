"""Pit-first terrain for V54: one extra, easier rung *below* the stock curriculum.

The extra rung is called level **-1**: the stock curriculum keeps its levels 0-9, and the
pit sits one below its easiest row.  Only the *label* is negative -- ``terrain_levels``
still holds the grid row (0 .. ``PIT_ROWS - 1`` for the pit) because it indexes
``terrain_origins[row, col]``; ``pit_display_levels`` maps the row to the level number used
in logs and in this docstring.

The 6-sub-terrain grid is left untouched.  What changes is only the env allocation: a block
of ``PIT_ROWS`` rows is prepended to the grid and filled with the cheapest terrain V54 has,
IsaacLab's inverted-pyramid stairs at difficulty 0 (6 risers of 50 mm, centre platform
0.35 m below the rim).  ``max_init_terrain_level`` then puts every env in that block, and
``terrain_levels_pit_first`` promotes an env out of level -1 straight into row ``PIT_ROWS``
-- stock level 0 -- once it covers 4 m, i.e. a per-env warm-up instead of a fixed iteration
cut-off.  Promoted envs are floored at level 0: nothing ever walks back into the pit.

Rows ``PIT_ROWS .. PIT_ROWS + 9`` are stock levels 0-9, row for row: row ``PIT_ROWS + r``
is generated with ``difficulty = (r + U(0,1)) / 10``, the value stock level ``r`` has today.

Why a whole block of rows instead of a single pit row: the envs are spread over the grid by
``TerrainImporter._compute_env_origins_curriculum`` (level = row, type = column), so one pit
row would hold all 4096 envs on its 20 tiles -- 205 per 8x8 m tile, ten times the usual
spacing, with the articulations spawning inside each other.  ``PIT_ROWS`` 10 x 20 columns
gives 200 pit tiles, which keeps the density at the ~20 env/tile the stock grid runs at.
"""
from __future__ import annotations

import copy
import traceback

import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGenerator

PIT_ROWS = 10
"""Rows of the grid that are the pit.  ``num_rows - PIT_ROWS`` rows carry the stock terrain."""

PIT_LEVEL = -1
"""Curriculum level number of the pit.  Stock levels 0-9 are rows ``PIT_ROWS + 0 .. 9``."""

PIT_STEP_HEIGHT = 0.05
"""Riser height of the pit in m: ``pyramid_stairs_inv`` at difficulty 0."""


def pit_display_levels(levels: torch.Tensor) -> torch.Tensor:
    """Grid rows -> curriculum level numbers: pit rows become -1, stock rows keep 0-9.

    ``terrain_levels`` has to stay a row index (it is used as ``terrain_origins[row, col]``,
    so a negative value would wrap to the far end of the grid).  Everything that is read by
    a human -- the ``Curriculum/terrain_levels`` curve above all -- goes through here, so the
    extra rung shows up as level -1 instead of as a second block of rows 0-9.
    """
    return torch.where(levels < PIT_ROWS, torch.full_like(levels, PIT_LEVEL), levels - PIT_ROWS)


class PitFirstTerrainGenerator(TerrainGenerator):
    """``TerrainGenerator`` whose first ``PIT_ROWS`` rows are the 50 mm inverted stair pit."""

    def __init__(self, cfg, device: str = "cpu"):
        # The pit config has to exist before ``super().__init__`` -- that call is what
        # generates the meshes, and ``_generate_curriculum_terrains`` reads ``self.pit_cfg``.
        # It is not an entry of ``cfg.sub_terrains`` (those are laid out by column and share
        # one difficulty ramp), so ``size`` is copied in by hand; the stock constructor only
        # propagates it to the sub-terrains it knows about.
        pit = copy.deepcopy(cfg.sub_terrains["pyramid_stairs_inv"])
        pit.proportion = 1.0
        pit.step_height_range = (PIT_STEP_HEIGHT, PIT_STEP_HEIGHT)
        pit.size = cfg.size
        self.pit_cfg = pit
        super().__init__(cfg, device)

    def _generate_curriculum_terrains(self):
        """``TerrainGenerator._generate_curriculum_terrains`` with the pit rows in front.

        The stock layout rule is kept for the stock rows: the column index picks the
        sub-terrain type (through ``proportion``), the row index picks the difficulty.
        """
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_indices = np.array(
            [
                int(np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))
                for index in range(self.cfg.num_cols)
            ],
            dtype=np.int32,
        )
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        lower, upper = self.cfg.difficulty_range
        stock_rows = self.cfg.num_rows - PIT_ROWS

        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                if sub_row < PIT_ROWS:
                    # Difficulty 0 every time, and no ``np_rng`` draw: the pit rows are
                    # identical copies that exist only to hold the envs apart.
                    mesh, origin = self._get_terrain_mesh(0.0, self.pit_cfg)
                    self._add_sub_terrain(mesh, origin, sub_row, sub_col, self.pit_cfg)
                    continue
                sub_cfg = sub_terrains_cfgs[sub_indices[sub_col]]
                difficulty = (sub_row - PIT_ROWS + self.np_rng.uniform()) / stock_rows
                difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)


def terrain_type_names(terrain) -> list[str]:
    """Sub-terrain name of every grid column.

    ``TerrainGenerator._generate_terrains`` does not sample the sub-terrains randomly for a
    fixed grid: column ``c`` goes to
    ``argmin(c / num_cols + 0.001 < cumsum(normalised proportions))``.  Rebuilding that
    expression is what lets the curriculum log a curve per sub-terrain
    (``Curriculum/by_type/<metric>/<name>``, see ``episode_stats_by_terrain_type``) instead of
    only the batch mean.

    On the V54 grid (``ROUGH_TERRAINS_CFG``, 20 columns) the mapping is
    ``pyramid_stairs`` 0-3, ``pyramid_stairs_inv`` 4-7, ``boxes`` 8-11, ``random_rough``
    12-15, ``hf_pyramid_slope`` 16-17, ``hf_pyramid_slope_inv`` 18-19.
    """
    gen_cfg = terrain.cfg.terrain_generator
    names = list(gen_cfg.sub_terrains.keys())
    proportions = np.array(
        [sub_cfg.proportion for sub_cfg in gen_cfg.sub_terrains.values()], dtype=np.float64
    )
    proportions /= np.sum(proportions)
    cumulative = np.cumsum(proportions)
    return [
        names[int(np.min(np.where(col / gen_cfg.num_cols + 0.001 < cumulative)[0]))]
        for col in range(gen_cfg.num_cols)
    ]


def type_masks(terrain) -> dict[str, torch.Tensor]:
    """One boolean mask per sub-terrain type, over the env axis.

    ``terrain.terrain_types`` is the env's *column* index, so the column -> name mapping from
    :func:`terrain_type_names` is what turns it into "this env is on pyramid_stairs".
    """
    names = terrain_type_names(terrain)
    terrain_types = terrain.terrain_types
    masks: dict[str, torch.Tensor] = {}
    for name in dict.fromkeys(names):
        mask = torch.zeros_like(terrain_types, dtype=torch.bool)
        for col, col_name in enumerate(names):
            if col_name == name:
                mask |= terrain_types == col
        masks[name] = mask
    return masks


def levels_by_terrain_type(terrain, levels: torch.Tensor) -> dict[str, float]:
    """Mean of ``levels`` per sub-terrain type, ready for a dict-returning log term."""
    return {
        name: levels[mask].float().mean().item()
        for name, mask in type_masks(terrain).items()
        if bool(mask.any())
    }


TYPE_STATS_TERMINATIONS = {
    "fell_over": "fall",
    "time_out": "timeout",
    "dof_pos_limit": "dof",
    "base_contact": "base",
}
"""Termination term -> short key in the per-type log (missing terms are simply skipped)."""


def episode_stats_by_terrain_type(env, env_ids, display_levels: bool = True) -> dict[str, float]:
    """How the episodes that just finished performed, split by sub-terrain type.

    A batch mean cannot say *which* terrain a run is failing on, so this returns one key per
    (metric, type) pair -- ``Curriculum/by_type/<metric>/<sub-terrain>``.  Metrics: ``n``
    finished episodes, ``level`` (the row they were played on), ``len`` policy steps
    survived, ``ret`` episode reward, ``dist`` net displacement from the spawn point (the 4 m
    promotion statistic) and one rate per termination term.

    Read *before* the reset: ``ManagerBasedRLEnv._reset_idx`` runs the curriculum managers
    first and only then resets the scene, so ``episode_length_buf``,
    ``RewardManager._episode_sums`` and ``TerminationManager._last_episode_dones`` still hold
    the finished episode.  The term is also registered ahead of ``terrain_levels``, because
    promotion rewrites ``env_origins`` -- the origin ``dist`` is measured against.

    Buckets: one per sub-terrain, plus ``pit`` for the level -1 rows of the pit-first grid.
    The pit rows are laid out by column too, so without the split they would be counted under
    a terrain type they are not.

    Rates, not counts, and only over finished episodes: the sub-terrain proportions differ
    (4/4/4/4/2/2 columns on the 20-column grid) and a per-step mean over all envs would drown
    the signal in the ~99 % of envs that are mid-episode.

    Caveat when reading the curves: the curriculum promotes each type as *it* succeeds, so a
    type whose envs climbed faster is also playing harder rows -- compare the rates together
    with ``level``.
    """
    terrain = env.scene.terrain
    try:
        if int(getattr(env, "common_step_counter", 2)) <= 1:
            # Every episode that ends on the very first policy step ended on the pre-placement
            # grid, not on its tile (see ``terrain_levels_pit_first``): logging it would put a
            # 20-100 m spike into ``dist`` on the first iteration.
            return {}
        rows = terrain.terrain_levels
        levels = pit_display_levels(rows) if display_levels else rows
        if isinstance(env_ids, slice):
            finished = torch.arange(levels.shape[0], device=levels.device)
        else:
            finished = torch.as_tensor(env_ids, device=levels.device, dtype=torch.long).flatten()

        # End-of-episode state, still intact here (see docstring).
        ep_len = env.episode_length_buf[finished].float()
        ep_ret = torch.zeros(finished.numel(), dtype=torch.float, device=levels.device)
        for sums in env.reward_manager._episode_sums.values():
            ep_ret += sums[finished].float()
        offset = env.scene["robot"].data.root_pos_w[finished, :2] - env.scene.env_origins[finished, :2]
        ep_dist = torch.norm(offset, dim=-1)

        termination = env.termination_manager
        term_rates = [
            (alias, termination._last_episode_dones[finished, termination.active_terms.index(name)].float())
            for name, alias in TYPE_STATS_TERMINATIONS.items()
            if name in termination.active_terms
        ]

        buckets = {name: mask & (levels >= 0) for name, mask in type_masks(terrain).items()}
        if bool((levels < 0).any()):
            buckets["pit"] = levels < 0
        selection = {name: mask[finished] for name, mask in buckets.items()}
        # One sync for the empty check, one for the stats themselves: this runs on every reset.
        counts = torch.stack([sel.sum() for sel in selection.values()]).tolist()

        keys: list[str] = []
        values: list[torch.Tensor] = []
        for (name, sel), count in zip(selection.items(), counts):
            if count == 0:
                continue
            ids = finished[sel]
            keys += [f"n/{name}", f"level/{name}", f"len/{name}", f"ret/{name}", f"dist/{name}"]
            values += [
                torch.as_tensor(float(count), device=levels.device),
                levels[ids].float().mean(),
                ep_len[sel].mean(),
                ep_ret[sel].mean(),
                ep_dist[sel].mean(),
            ]
            for alias, rate in term_rates:
                keys.append(f"{alias}/{name}")
                values.append(rate[sel].mean())
        return dict(zip(keys, torch.stack(values).tolist()))
    except Exception:  # noqa: BLE001 - logging must never take a 30 h training run down
        if not getattr(terrain, "_by_type_broken", False):
            terrain._by_type_broken = True
            print("[WARN] per-terrain-type logging disabled:", flush=True)
            traceback.print_exc()
        return {}


def log_episode_stats_stock(env, env_ids) -> dict[str, float]:
    """``Curriculum/by_type/...`` for the stock 10-row grid (no pit rows to split out)."""
    return episode_stats_by_terrain_type(env, env_ids, display_levels=False)


def log_episode_stats_pit(env, env_ids) -> dict[str, float]:
    """Same, in display levels (pit rows count as -1) for the pit-first grid."""
    return episode_stats_by_terrain_type(env, env_ids, display_levels=True)


def _stock_move_flags(
    env, env_ids, asset_cfg: SceneEntityCfg, ran: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two tests of ``mdp.terrain_levels_vel``, with the t=0 guard folded in.

    * ``move_up``   : ``distance > size[0] / 2`` -- 4 m of *net* displacement from the spawn
                      point (the tile centre).
    * ``move_down`` : ``distance < |v_cmd_xy| * max_episode_length_s * 0.5`` -- the stock rule,
                      i.e. less than half of the displacement the command asked for (5 m at
                      ``vx = 1 m/s`` on the 10 s episodes).  Only the x/y command enters;
                      ``ang_vel_z`` is not part of the norm.

    Kept here instead of in ``isaaclab_tasks`` because neither term is usable unchanged: the
    stock version judges envs that have not taken a step yet, and it re-places envs that clear
    the last row on a *random* row.  See ``terrain_levels_pit_first`` for the t=0 artefact.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = (distance > terrain.cfg.terrain_generator.size[0] / 2) & ran
    move_down = (
        distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    ) & ran
    # the stock code does ``move_down *= ~move_up``: walking 4 m is always a promotion, even if
    # the command asked for more than that
    move_down = move_down & ~move_up
    return move_up, move_down


def terrain_levels_pit_first(
    env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``mdp.terrain_levels_vel`` with level -1 as a one-way floor and no random re-placement.

    * An env at level -1 (row < ``PIT_ROWS``) that walks ``size[0] / 2`` = 4 m jumps straight
      to stock level 0 (row ``PIT_ROWS``).  It does not climb the pit block one row per
      episode: those rows are identical, so that would only waste episodes.
    * An env on the stock terrain moves one row up or down with the *stock* tests (see
      ``_stock_move_flags``), with level 0 as the floor: an env that fails an episode never
      drops back into the pit, those rows are cleared and re-doing them only costs episodes.
    * The last row is where the curriculum *closes*: an env that reaches it stays there (no
      random re-placement), so the hardest terrain is the steady state instead of a
      one-episode visit.  This is the only difference to the stock motion.

    The returned mean is in level numbers (``pit_display_levels``), so a run that starts with
    every env in the pit logs ``Curriculum/terrain_levels`` at -1 and climbs towards 9.

    Envs that have not stepped yet are not judged at all: at t=0 the term runs before the
    robots are spawned, so their positions say nothing about how far they walked (see the
    ``ran`` mask below).
    """
    terrain = env.scene.terrain
    # ``_reset_idx`` calls this *before* ``scene.reset()`` and the reset events, so on the very
    # first call (t=0) the robots are still at their pre-spawn state and ``root_pos_w`` has no
    # relation to ``env_origins``: the measured distance is 6-109 m for every env, i.e. "walked
    # 4 m" is true for all 4096 and the whole pit empties before it is ever used (level -1 is
    # then unreachable, because promotion out of it is one-way).  An env that has not taken a
    # step yet cannot have walked anywhere, so only envs with a non-zero episode length are
    # judged -- at t=0 that is nobody, and afterwards it is exactly the envs whose episode just
    # ended.
    #
    # ``episode_length_buf > 0`` alone is not enough: ``SataControlFreqEnv._sata_substep_step``
    # bumps it once *before* the substep loop, so the very first policy step already looks like a
    # finished episode -- while the robots are still on the pre-placement grid (the ``scene.reset``
    # that puts them on their tiles happens at the end of that same step).  That is why the
    # 2026-10-24_21-46-46 run left the pit at step 0 and only ever logged level -1 for the
    # pre-step entry.  Nothing can have walked 4 m within the first policy step of its life, so
    # that step is never judged either.
    ran = (env.episode_length_buf[env_ids] > 0) & (env.common_step_counter > 1)
    # ``move_up`` / ``move_down`` are the stock tests; ``ran`` keeps the t=0 artefact out (the
    # pre-step distance is 6-109 m, which would promote the whole batch out of the pit).
    move_up, move_down = _stock_move_flags(env, env_ids, asset_cfg, ran)

    levels = terrain.terrain_levels[env_ids]
    in_pit = levels < PIT_ROWS
    # out of the pit: one jump, and only upwards
    pit_next = torch.where(move_up, torch.full_like(levels, PIT_ROWS), levels)
    # on the stock rows: stock uphill/downhill motion between level 0 (row ``PIT_ROWS``) and the
    # last row.  ``max_terrain_level`` is ``num_rows``
    # (``TerrainImporter._compute_env_origins_curriculum``) while ``terrain_origins`` is indexed
    # 0 .. num_rows - 1, so the fixed point has to be one *below* it: clamping at
    # ``max_terrain_level`` itself would index one row past the grid and raise.  Clamping at
    # ``top_row`` is also what freezes an env that clears the last row.  The floor at
    # ``PIT_ROWS`` keeps demotions out of the pit.
    top_row = int(terrain.max_terrain_level) - 1
    stock_next = torch.clamp(
        levels + move_up.long() - move_down.long(), min=PIT_ROWS, max=top_row
    )
    new_levels = torch.where(in_pit, pit_next, stock_next)

    terrain.terrain_levels[env_ids] = new_levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        new_levels, terrain.terrain_types[env_ids]
    ]
    return torch.mean(pit_display_levels(terrain.terrain_levels).float())


def terrain_levels_top_freeze(
    env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Stock uphill *and* downhill motion, except that the last row is a dead end (V54 rev G).

    The climb is the stock one: 4 m of net displacement promotes an env by one row, and less
    than half of the commanded displacement demotes it by one row (``_stock_move_flags``).
    Demotion stays in on purpose -- it is the part that keeps an env from sitting on terrain it
    cannot actually handle, and with the yaw range gated on the level
    (``SataVelocityCommandCfg.level_scaled_yaw``) a low-level episode is mostly straight
    walking, so the demotion test no longer fires on turn-heavy commands.

    The difference to ``mdp.terrain_levels_vel`` is the top: stock re-places an env that clears
    the last row on a *random* row (``TerrainImporter.update_env_origins``), which throws the
    climb away exactly for the envs that earned it.  Here the last row is a fixed point, so
    reaching the top closes the curriculum for that env and the hardest terrain -- together
    with the full SATA turn rate, since ``level_scaled_yaw`` reaches 1.0 at ``level_ref`` --
    becomes its steady state.

    The ``ran`` mask is the same t=0 guard as ``terrain_levels_pit_first``: this term runs
    before the robots are spawned (and, on the first policy step, before ``scene.reset`` puts
    them on their tiles), so pre-spawn positions would move the entire batch.
    """
    terrain = env.scene.terrain
    ran = (env.episode_length_buf[env_ids] > 0) & (env.common_step_counter > 1)
    move_up, move_down = _stock_move_flags(env, env_ids, asset_cfg, ran)

    # ``max_terrain_level`` is ``num_rows`` while ``terrain_origins`` is indexed 0 .. num_rows-1,
    # so clamp one row below it (see ``terrain_levels_pit_first``).  ``max=top_row`` is what
    # freezes the top: a promotion on the last row clamps straight back onto it.
    top_row = int(terrain.max_terrain_level) - 1
    new_levels = torch.clamp(
        terrain.terrain_levels[env_ids] + move_up.long() - move_down.long(),
        min=0,
        max=top_row,
    )
    terrain.terrain_levels[env_ids] = new_levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        new_levels, terrain.terrain_types[env_ids]
    ]
    return torch.mean(terrain.terrain_levels.float())
