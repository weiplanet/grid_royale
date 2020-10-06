from __future__ import annotations

from typing import Optional

import gamey


class GameyException(Exception):
    pass

class IllegalAction(GameyException):
    def __init__(self, action: Optional[gamey.Action] = None) -> None:
        self.action = action
        GameyException.__init__(self, f"Can't play {action} in this state.")


