from dataclasses import dataclass, field
from typing import Any


### This class aims to transport the information about a detected object. It's called Temporary Person, as it's exist for only ONE frame. After that, it's expected that it's atributes change.

@dataclass
class TempPerson:
    __id : int = 0 ### Keep in mind ID's should NOT repeat. However, there is no overwriting in that. Not until the REID system. So, to simplificate the creation of the class, we can use a default value without trouble
    __bb : list = field(default_factory=list)
    __confidence : float = 0
    """ __position   : dict = field(default_factory=dict) """
    
    @property
    def id(self):
        return self.__id
    @id.setter
    def id(self, value):
        self.__id = value
    
    @property
    def bb(self):
        return self.__bb
    @bb.setter
    def bb(self, value):
        self.__bb = value
        
    @property
    def confidence(self):
        return self.__confidence
    @confidence.setter
    def confidence(self, value):
        self.__confidence = value
    
    """ Discarted atribute
     
    @property
    def position(self):
        return self.__position
    @position.setter
    def position(self, value):
        self.__position = value
        """