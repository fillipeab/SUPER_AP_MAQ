from dataclasses import dataclass, field
from typing import Any


### This class aims to transport the information about a detected object. It's called Temporary Person, as it's exist for only ONE frame. After that, it's expected that it's atributes change.

@dataclass
class TempPerson:
    id : int = 0 ### Keep in mind ID's should NOT repeat. However, there is no overwriting in that. Not until the REID system. So, to simplificate the creation of the class, we can use a default value without trouble
    bb : list = field(default_factory=list)
    confidence : float = 0