from dataclasses import dataclass, field
from typing import Any
from PersonDB import PersonDB

@dataclass
class MemorySystem:
    ### List that stores all DB's. Initially, we might expect to work with just one DB. However, there is no practical reason to limit our methods to that.
    person_DBs : list[PersonDB] = field(default_factory=list)
    
    ###Person_DBs methods(add, remove by pos)
    def add_person_DBs(self,new_db):
        self.person_DBs.append(new_db)
    
    def remove_person_DBs(self,pos=-1):
        self.person_DBs.pop(pos)
    
    def get_person_DBs(self,pos=0):
        return self.person_DBs[pos]
        
    def __call__(self):
        person_db_standard = PersonDB()
    
        
    
    
    