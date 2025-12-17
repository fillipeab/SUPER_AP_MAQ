from dataclasses import dataclass, field
from typing import Any
from PersonDB import PersonDB

@dataclass
class MemorySystem:
    ### List that stores all DB's. Initially, we might expect to work with just one DB. However, there is no practical reason to limit our methods to that.
    person_DBs : list[PersonDB] = field(default_factory=list)
    reid_systems : list[Any] = field(default_factory=list)
    
    def __post_innit__(self):
        self.person_DBs.append(PersonDB())
    
    ###Re-id System list methods(add, remove by pos)
    def add_reid_system(self,new_system):
        self.reid_systems.append(new_system)
    
    def remove_reid_system(self,pos=-1):
        self.reid_systems.pop(pos)
        
    ###Person_DBs methods(add, remove by pos)
    def add_person_DBs(self,new_db):
        self.person_DBs.append(new_db)
    
    def remove_person_DBs(self,pos=-1):
        self.person_DBs.pop(pos)
    
    def get_person_DBs(self,pos=0):
        return self.person_DBs[pos]
        
    
    ###The object is, by definition, the output of the identification system(probably YOLO), in the format of temporary_person list, and the output of the method will be the best-matched id, if there is one, for each. Especifically, the temporary_person list will change the temporary id for a new one.
    ### By definition, there should be at least 1 reid and one person_db. More can be added.
    def reid_analyse(self,object,reid_pos=0,person_db_pos=0):
        reid_result = self.reid_systems[reid_pos](person_db_pos,object)
        return reid_result
    
    
    
    