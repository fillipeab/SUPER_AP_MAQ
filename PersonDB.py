from dataclasses import dataclass, field
from typing import Any
from Person import Person

@dataclass
class PersonDB:
    stored_persons : dict = field(default_factory=dict)
    
    def add(self, item : Person):
        self.stored_persons[Person.id] = Person
    
    def remove(self, pos : int = -1):
        self.stored_persons[pos].pop(pos)
    
    @property
    def size(self):
        return len(self.stored_persons)
    
    def get_person_by_id(self, id : int = 0):
        return self.stored_persons[id]
            