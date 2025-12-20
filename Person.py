from dataclasses import dataclass, field
from typing import Any

@dataclass
class Person:
    id : int = 0
    features : list[Any] = field(default_factory=list)

           
    