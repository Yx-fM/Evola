"""Event bus for loose coupling between modules.

08_kunyu publishes events, 09_junjian subscribes and logs/renders.
No module needs to import another module directly.
"""

from typing import Callable, Any
from collections import defaultdict


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[[dict[str, Any]], None]):
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: dict[str, Any]):
        for callback in self._subscribers.get(event_type, []):
            callback(data)

    def clear(self):
        self._subscribers.clear()
