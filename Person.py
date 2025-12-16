from dataclasses import dataclass
from typing import any

@dataclass
class Person:
    id : int
    last_seen : any
    seen_count : int
    Features : list