from dataclasses import dataclass, field
from typing import Any
from Person import Person

@dataclass
class PersonDB:
    stored_persons : dict = field(default_factory=dict)
    
    def add(self, new_person : Person):
        self.stored_persons[new_person.id] = new_person
    
    @property
    def size(self):
        return len(self.stored_persons)
    
    def get_person_by_id(self, id : int = 0):
        if id in self.stored_persons: ###Makes sure the id is present
            return self.stored_persons[id]
        else:
            return None
            