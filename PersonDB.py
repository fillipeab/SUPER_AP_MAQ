from dataclasses import dataclass, field
from typing import Any
from Person import Person

@dataclass
class PersonDB:
    stored_people : dict = field(default_factory=dict)
    
    def add(self, new_person : Person):
        self.stored_people[new_person.id] = new_person
    
    @property
    def size(self):
        return len(self.stored_people)
    
    def get_person_by_id(self, id : int = 0):
        if id in self.stored_people: ###Makes sure the id is present
            return self.stored_people[id]
        else:
            return None
            