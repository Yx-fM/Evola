"""JunJian logger - subscribes to EventBus, writes JSON Lines."""

import json
import os
from datetime import datetime
from enum import Enum

import numpy as np


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class PrivacyLevel(Enum):
    PUBLIC = 0
    DEBUG = 1


class JunJian:
    def __init__(self, bus, log_dir: str = "logs", privacy: PrivacyLevel = PrivacyLevel.PUBLIC):
        self.bus = bus
        self.privacy = privacy
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"evola_{timestamp}.jsonl")
        self._log_file = open(self.log_path, "w", encoding="utf-8")
        self._subscribed = {}

    def start(self):
        self._subscribe("step.begin", self._on_step)
        self._subscribe("step.outcome", self._on_outcome)
        self._subscribe("agent.decision", self._on_decision)
        return self

    def stop(self):
        for event_type in self._subscribed:
            for cb in self._subscribed[event_type]:
                try:
                    self.bus._subscribers[event_type].remove(cb)
                except (ValueError, AttributeError):
                    pass
        self._log_file.close()

    def _subscribe(self, event_type, callback):
        self.bus.subscribe(event_type, callback)
        self._subscribed.setdefault(event_type, []).append(callback)

    def _write(self, entry: dict):
        self._log_file.write(json.dumps(_to_native(entry), ensure_ascii=False) + "\n")
        self._log_file.flush()

    def _on_step(self, data):
        self._write({"type": "step.begin", **data})

    def _on_outcome(self, data):
        self._write({"type": "step.outcome", **data})

    def _on_decision(self, data):
        if self.privacy == PrivacyLevel.PUBLIC:
            data = {k: v for k, v in data.items() if not k.startswith("_")}
        self._write({"type": "agent.decision", **data})

    def get_log_path(self):
        return self.log_path

    def get_log_entries(self):
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))
        return entries
