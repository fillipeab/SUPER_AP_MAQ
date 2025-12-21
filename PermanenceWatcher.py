### Process 1, phase 2: find permanence
from dataclasses import dataclass, field
from TempPerson import TempPerson

"""
Input: List[Temp_person]
Memory: dict with Temp_person
Output: List of permanent people
"""

@dataclass
class PermanenceWatcher():
    ### PermanenceWatcher[list[TempPerson]] -> list[TempPerson]
    permanent_people_counter_dict : dict[int,int] = field(default_factory=dict) ###dict with key -> times seen
    permanent_people_dict : dict[int,TempPerson] = field(default_factory=dict)
    LIMIT_SEEN_COUNTER = 100
    EXPORT_THRESHOLD = 5
    PERMANENT_THRESHOLD = 5
    OUT_OF_PERMANENCE_THRESHOLD = PERMANENT_THRESHOLD-3 ###must be bigger than discart THRESHOLD
    DISCART_THRESHOLD = -30
    
    def __post_init__(self):
        if self.DISCART_THRESHOLD>self.OUT_OF_PERMANENCE_THRESHOLD:
            self.OUT_OF_PERMANENCE_THRESHOLD=self.DISCART_THRESHOLD
    
    
    def __call__(self,list_of_temporary_people : list[TempPerson]) -> list[TempPerson]: ###KEEEP IN MIND, THIS LIST HAS ALREADY BEEN PROCESSED BY PHASE 1
        p_pc_dict = self.permanent_people_counter_dict
        p_people_dict = self.permanent_people_dict
        return_list = []
        
        ### part 1 - update the entry ###
        for temp_person in list_of_temporary_people:
            tp_id = int(temp_person.id)
            if tp_id in p_pc_dict: ###checks if id is registred in dict
                
                ### debug ####  print("old",tp_id,"contando",p_pc_dict[tp_id])
                
                ###Deals with Discart_threshold###
                if p_pc_dict[tp_id]<0: 
                    p_pc_dict[tp_id]=0
                p_pc_dict[tp_id] += 2 ###important
                
                ### UPDATE PERMANENT PERSON DICT - used here to avoid 2 for checking, without need ###
                if tp_id in p_people_dict: ###already inside permanent people dict
                    p_people_dict[tp_id] = temp_person ###UPDATES INFO in dict
                    if p_pc_dict[tp_id] > self.EXPORT_THRESHOLD: ### Is this person seen count above exporting threshold? ###ONLY EXPORTS INFO OF PEOPLE THAT HAVE BEEN SEEN IN FRAME!
                        return_list.append(p_people_dict[tp_id]) ###Yes: Adds to export list
                elif p_pc_dict[tp_id]-1 > self.PERMANENT_THRESHOLD: ###-1 to account for the +2 previously added
                    p_people_dict[tp_id] = temp_person ###Updates the person
                
                    
                    ### debug ### print("new_permanent_one")
                ### end ###
            ###not in dict - add to id ###    
            else:
                p_pc_dict[tp_id] = 2
            ### end ###
        ### part 1 - end ###        
        
        ### part 2 - update memory
        remove_from_p_pc_dict = []
        for key in p_pc_dict:
            p_pc_dict[key]-=1
            ### Maximum limit
            if p_pc_dict[key] > self.LIMIT_SEEN_COUNTER:
                p_pc_dict[key] = self.LIMIT_SEEN_COUNTER
            ### Minimum limit
            if p_pc_dict[key] < self.DISCART_THRESHOLD: ###Remove from memory person absent for too long - avoids memory overflow
                remove_from_p_pc_dict.append(key)
                ###debug###
                ### print("BANNED")
                ### print(key,p_pc_dict[key])
                
        ##remove from dict##
        for key in remove_from_p_pc_dict:
            p_pc_dict.pop(key,None)
        ### part 2 - end ###
        
        ### part 3 - create an export list of permanent person and remove "forgotten" people
        remove_from_p_person_dict = []
        for key in p_people_dict:
            if p_pc_dict[key] < self.OUT_OF_PERMANENCE_THRESHOLD: ###Person absent for too long
                remove_from_p_person_dict.append(key)
        for key in remove_from_p_person_dict:
            p_people_dict.pop(key,None)
        ### part 3 - end ###
        return return_list