from __future__ import annotations

from typing import Optional

from .base import Action


class GameyException(Exception):
    pass

class IllegalAction(GameyException):
    def __init__(self, action: Optional[Action] = None) -> None:
        self.action = action or "the given action"
        GameyException.__init__(self, f"Can't play {action} in this state.")


