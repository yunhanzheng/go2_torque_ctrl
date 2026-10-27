"""SATA ``go2_torque`` training terrain, ported to IsaacLab height fields.

``go2_torque_config.py:66-90`` trains on ``mesh_type = 'trimesh'`` with
``terrain_proportions = [0.2, 0.8, 0, 0, 0]`` and ``curriculum = False``, so
``legged_gym/utils/terrain.py::Terrain.randomized_terrain`` draws every one of the
10 x 20 tiles independently as

  type        : 20 % smooth slope / 80 % rough slope
  difficulty  : ``np.random.choice([0.5, 0.75, 0.9])``
  slope       : ``0.1 * difficulty`` (5 % / 7.5 % / 9 %, i.e. 2.9 / 4.3 / 5.1 deg),
                random sign, truncated pyramid with a 3 m platform
  roughness   : rough slopes add ``random_uniform_terrain`` noise of +-6 cm sampled on a
                0.2 m grid (5 mm height steps)

``play.py:57`` is the only place that switches to ``mesh_type = 'plane'``; training never
sees flat ground.

The two numpy helpers below are line-by-line ports of
``isaacgym.terrain_utils.pyramid_sloped_terrain`` / ``random_uniform_terrain``.  The only
substitution is ``scipy.interpolate.interp2d`` (removed in modern scipy) ->
``RectBivariateSpline(kx=1, ky=1)``, which is the interpolator ``interp2d(kind="linear")``
used internally.

The generator's own ``difficulty`` argument is ignored on purpose: SATA samples the
difficulty per tile from a discrete set, which a ``difficulty_range`` cannot express.
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


def _pyramid_sloped(hf: np.ndarray, h_scale: float, v_scale: float, slope: float, platform_width: float) -> None:
    """``terrain_utils.pyramid_sloped_terrain`` (Isaac Gym), in place on ``hf``."""
    width, length = hf.shape
    center_x, center_y = width // 2, length // 2
    xx, yy = np.meshgrid(np.arange(width), np.arange(length), sparse=True)
    xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width, 1)
    yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length)
    max_height = int(slope * (h_scale / v_scale) * (width / 2))
    hf += (max_height * xx * yy).astype(hf.dtype)

    # flatten the central platform to the height of its own corner (Isaac Gym's clip)
    p = int(platform_width / h_scale / 2)
    x1, x2 = width // 2 - p, width // 2 + p
    y1, y2 = length // 2 - p, length // 2 + p
    np.clip(hf, min(hf[x1, y1], 0), max(hf[x1, y1], 0), out=hf)


def _random_uniform(
    hf: np.ndarray,
    h_scale: float,
    v_scale: float,
    noise_range: tuple[float, float],
    noise_step: float,
    downsampled_scale: float,
) -> None:
    """``terrain_utils.random_uniform_terrain`` (Isaac Gym), in place on ``hf``."""
    width, length = hf.shape
    # number of coarse samples: int(size / downsampled_scale), as in Isaac Gym
    n_x = int(width * h_scale / downsampled_scale)
    n_y = int(length * h_scale / downsampled_scale)
    height_min = int(noise_range[0] / v_scale)
    height_max = int(noise_range[1] / v_scale)
    height_step = int(noise_step / v_scale)
    heights = np.arange(height_min, height_max + height_step, height_step)
    coarse = np.random.choice(heights, size=(n_x, n_y)).astype(np.float64)

    span_x = width * h_scale
    span_y = length * h_scale
    spline = interpolate.RectBivariateSpline(
        np.linspace(0.0, span_x, n_x), np.linspace(0.0, span_y, n_y), coarse, kx=1, ky=1
    )
    upsampled = spline(np.linspace(0.0, span_x, width), np.linspace(0.0, span_y, length))
    hf += np.rint(upsampled).astype(hf.dtype)


@height_field_to_mesh
def sata_sloped_terrain(difficulty: float, cfg: SataSlopedTerrainCfg) -> np.ndarray:
    """One ``make_terrain`` branch: pyramid slope, optionally with uniform roughness."""
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((width, length), dtype=np.int16)

    # SATA draws the difficulty per tile; the generator-provided value is unused.
    slope = cfg.slope_scale * float(np.random.choice(cfg.difficulty_choices))
    if np.random.uniform() < cfg.negative_prob:
        slope = -slope
    _pyramid_sloped(hf, cfg.horizontal_scale, cfg.vertical_scale, slope, cfg.platform_width)

    if cfg.rough:
        _random_uniform(
            hf,
            cfg.horizontal_scale,
            cfg.vertical_scale,
            cfg.noise_range,
            cfg.noise_step,
            cfg.downsampled_scale,
        )
    return hf


@configclass
class SataSlopedTerrainCfg(HfTerrainBaseCfg):
    """SATA's smooth-slope (20 %) and rough-slope (80 %) sub-terrains."""

    function = sata_sloped_terrain

    difficulty_choices: tuple[float, ...] = (0.5, 0.75, 0.9)
    """``np.random.choice([0.5, 0.75, 0.9])`` in ``Terrain.randomized_terrain``."""

    slope_scale: float = 0.1
    """``slope = difficulty * 0.1`` in ``Terrain.make_terrain``."""

    platform_width: float = 3.0
    """``pyramid_sloped_terrain(..., platform_size=3.)``."""

    rough: bool = False
    """``True`` adds ``random_uniform_terrain`` on top of the slope."""

    negative_prob: float = 0.5
    """Probability of a downward (inverted) pyramid.

    SATA flips the sign when ``choice < proportions[i] / 2 (+ proportions[i-1] / 2)``, i.e.
    0.5 for the smooth branch and 0.3 / 0.8 = 0.375 for the rough branch.
    """

    noise_range: tuple[float, float] = (-0.06, 0.06)
    """``random_uniform_terrain(min_height=-0.06, max_height=0.06)``."""

    noise_step: float = 0.005
    """``random_uniform_terrain(step=0.005)``."""

    downsampled_scale: float = 0.2
    """``random_uniform_terrain(downsampled_scale=0.2)``."""
