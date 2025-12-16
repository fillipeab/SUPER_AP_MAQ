from dataclasses import dataclass, field
from typing import Any

@dataclass
class Person:
    id : int
    seen_count : int = 0
    Features : list[Any] = field(default_factory=list)