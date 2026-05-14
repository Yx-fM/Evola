"""MVP 1 — unified configuration (P0 + P1)."""

WORLD = {
    "height": 15, "width": 20, "vision_range": 2,
    "n_food": 8, "n_dangers": 2, "food_respawn_interval": 15,
    "max_food": 10, "wall_density": 0.03,
    "energy_per_food": 0.4, "damage_per_danger": 0.15,
}

HOMEOSTASIS = {
    "energy_initial": 0.8, "energy_decay": 0.003,
    "safety_rise": 0.8, "safety_decay": 0.05,
}

ARBITER = {
    "type": "ltc",            # "ltc" or "rule"
    "ltc_hidden": 64,
    "ltc_lr": 0.001,
    "use_self_model": True,
    # RuleArbiter fallback params (if type=="rule")
    "safety_threshold": 0.3,
    "energy_urgent": 0.5,
    "energy_safe": 0.3,
    "novelty_threshold": 0.2,
}

RUN = {
    "max_steps": 1000,
    "render_mode": "terminal",
    "delay": 0.05,
    "seed": 42,
    "save_mind": True,
    "save_stats": True,
    "save_log": True,
}
