from dataclasses import dataclass, field
from typing import Any

@dataclass
class Person:
    __id : int = 0
    __seen_count : int = 0
    __features : list[Any] = field(default_factory=list)
    
    @property
    def id(self): ###Allows to change the way the id is acessed
        return self.__id
    @id.setter
    def id(self, value):
        self.__id = value

    @property
    def seen_count(self):
        return self.__seen_count
    @seen_count.setter
    def seen_count(self, value):
        self.__seen_count = value 
        
    @property
    def features(self):
        return self.__features
    @features.setter
    def features(self, value):
        self.__features = value 

           
    