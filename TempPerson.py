from dataclasses import dataclass, field
from typing import Any


### This class aims to transport the information about a detected object. It's called Temporary Person, as it's exist for only ONE frame. After that, it's expected that it's atributes change.

@dataclass
Class TempPerson:
    __id : int = 0 ### Keep in mind ID's should NOT repeat. However, there is no overwriting in that. Not until the REID system. So, to simplificate the creation of the class, we can use a default value without trouble
    __BB : list = field(default_factory=list)
    __Confidence : float = 0
    
    @property
    def id(self):
        return self.__id
    @id.setter
    def id(self, value):
        self.__id = value
    
    @property
    def BB(self):
        return self.__BB
    @BB.setter
    def BB(self, value):
        self.__BB = value
        
    @property
    def Confidence(self):
        return self.__Confidence
    @Confidence.setter
    def Confidence(self, value):
        self.__Confidence = value