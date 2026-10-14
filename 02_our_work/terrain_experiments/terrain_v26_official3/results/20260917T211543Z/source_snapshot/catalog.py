"""Metadata that can be imported without launching Isaac Sim."""

NAMES = {
    "random_rough": "随机粗糙地面",
    "boxes": "随机方块／不规则台阶",
    "slope_up": "斜坡（上坡）",
    "slope_down": "斜坡（下坡）",
}

# Up/down slopes are two directions of the same terrain family.
FAMILIES = {
    "随机粗糙地面": ("random_rough",),
    "随机方块／不规则台阶": ("boxes",),
    "斜坡": ("slope_up", "slope_down"),
}

DIFFICULTIES = {"easy": 0.25, "hard": 0.75}
