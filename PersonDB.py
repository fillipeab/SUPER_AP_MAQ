from dataclasses import dataclass, field
from typing import Any
from Person import Person

@dataclass
class PersonDB:
    stored_persons : list[Person] = field(default_factory=list)
    
    def add(self, item : Person):
        self.stored_persons.append(item)
    
    def remove(self, pos : int = -1):
        self.stored_persons[pos].pop(pos)
    
    @property
    def size(self):
        return self.stored_persons.len()
    
    def get_person(self, pos : int = 0):
        return self.stored_persons[pos]
            