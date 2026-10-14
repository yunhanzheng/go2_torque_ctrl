"""Held-out evaluation cases for the four terrain families."""
CASES = {
    "plane": ("平地", 0.0),
    "random_rough": ("随机粗糙面（hard）", 0.75),
    "boxes": ("随机方块（hard）", 0.75),
    "slope_up": ("上坡（hard）", 0.75),
    "slope_down": ("下坡（hard）", 0.75),
    "stairs_up": ("10 cm上楼", 5.0 / 7.0),
    "stairs_down": ("10 cm下楼", 5.0 / 7.0),
}

