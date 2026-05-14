"""Evola Theme — 三实体三色盘

Kunyu (world)  : deep jade green  — earth, carrying, growth
Evola (agent)  : amber gold       — life, warmth, activity
JunJian (obs)  : distant slate    — calm, record, mirror

Auxiliary:
  privacy-line   : neutral mauve-grey
  debug-warning  : alert red (only when d is pressed)
  highlight      : pale gold (important events)
"""

# ── Primary palette ──
KUNYU = "rgb(42,111,92)"     # #2A6F5C  deep jade
EVOLA = "rgb(217,162,58)"    # #D9A23A  amber gold
JUNJIAN = "rgb(60,110,143)"  # #3C6E8F  distant slate

# ── Auxiliary ──
PRIVACY = "rgb(94,91,107)"   # #5E5B6B  neutral mauve
DEBUG = "rgb(214,92,92)"     # #D65C5C  alert red
HIGHLIGHT = "rgb(229,183,61)"  # #E5B73D  pale gold
DIM = "rgb(128,128,128)"     # dim grey

# ── Rich style strings ──
RICH_THEME = {
    "kunyu_border": KUNYU,
    "kunyu_title": KUNYU,
    "evola_color": EVOLA,
    "evola_bar": EVOLA,
    "evola_highlight": HIGHLIGHT,
    "junjian_color": JUNJIAN,
    "junjian_dim": DIM,
    "privacy_line": PRIVACY,
    "debug_border": DEBUG,
    "grid_wall": "rgb(60,80,60)",
    "grid_food": "bright_green",
    "grid_danger": "bright_red",
    "grid_agent": EVOLA,
    "grid_visited": "rgb(80,80,100)",
    "grid_empty": "rgb(40,40,50)",
    "event_move": "white",
    "event_eat": EVOLA,
    "event_danger": "bright_red",
    "event_explore": "rgb(180,200,180)",
    "event_calm": KUNYU,
}

# ── Color mapping for event types ──
EVENT_COLORS = {
    "move": "event_move",
    "explore": "event_explore",
    "eat": "event_eat",
    "danger_near": "event_danger",
    "damage": "event_danger",
    "calm": "event_calm",
}
