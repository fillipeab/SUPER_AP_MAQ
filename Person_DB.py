from dataclasses import dataclass
from typing import any
from Person import Person

@dataclass
class PersonDB:
    stored_persons : list[Person] = field(default_factory=list)
    
    def add(self, item : Person):
        self.stored_persons.append(item)
    
    def remove(self, pos : int = -1):
        if pos == -1:
            self.stored_persons[pos].pop()
        else:
            self.stored_persons[pos].pop(pos)
    
    @property
    def size(self):
        return self.stored_persons.len()
    
    def list_persons(self, pos : int = 0):
        return self.stored_persons[pos]
            