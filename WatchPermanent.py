### Processo 1 da fase 2 - achar fila
from dataclasses import dataclass, field
from typing import Any
from TempPerson import TempPerson

"""
Input: List[Temp_person]
Memory: dict with Temp_person
Output: List of permanent people
"""

@dataclass
class WatchPermanent():
    permanent_persons_counter_dict : dict[int,int] = field(default_factory=dict) ###dict with key -> times seen
    permanent_persons_dict : dict[int,TempPerson] = field(default_factory=dict)
    LIMIT_SEEN = 10;
    EXPORT_THRESHOLD = 5;
    PERMANENT_THRESHOLD = 5;
    OUT_OF_PERMANENCE_THRESHOLD = PERMANENT_THRESHOLD-3; ###must be bigger than discart THRESHOLD
    DISCART_THRESHOLD = -5;
    
    def __post_init__(self):
        if self.DISCART_THRESHOLD>self.OUT_OF_PERMANENCE_THRESHOLD:
            self.OUT_OF_PERMANENCE_THRESHOLD=self.DISCART_THRESHOLD
    
    
    def __call__(self,list_of_temporary_persons : list): ###KEEEP IN MIND, THIS LIST HAS ALREADY BEEN PROCESSED BY PHASE 1
        ### part 1 - update the entry ###
        p_pc_dict = self.permanent_persons_counter_dict
        p_persons_dict = self.permanent_persons_dict
        return_list = []
        
        
        for temp_person in list_of_temporary_persons:
            ### person already in dict
            tp_id = temp_person.id
            if temp_person.id in p_pc_dict:
                ###Deals with Discart_threshold###
                if p_pc_dict[tp_id]<0: 
                    p_pc_dict[tp_id]=0
                ### back to normal
                p_pc_dict[tp_id] += 2 ###important
                
                ### UPDATE PERMANENT PERSON DICT - used here to avoid 2 for checking, without need ###
                if p_pc_dict[tp_id]-1 > PERMANENT_THRESHOLD: ### -1 to account for the +2 previously added
                    p_persons_dict[tp_id] = temp_person ###Updates the person
                
                ### end ###
            ###not in dict - add to id ###    
            else: 
                p_pc_dict[temp_person.id] = 2
            ### end ###
        ### part 1 - end ###        
        
        ### part 2 - update memory
        for key in p_pc_dict:
            p_pc_dict[key]-=1
            ### Maximum limit
            if p_pc_dict[key] > self.LIMIT_SEEN:
                p_pc_dict[key] = LIMIT_SEEN
            ### Minimum limit
            if p_pc_dict[key] < -self.DISCART_THRESHOLD: ###Remove from memory person absent for too long - avoids memory overflow
                p_pc_dict.pop(key,None)
        ### part 2 - end ###
        
        ### part 3 - create an export list of permanent person and remove "forgotten" people
        for key in p_persons_dict:
            if p_persons_counter_dict[key] > self.EXPORT_THRESHOLD: ###Person seen enough times
                return_list.append(p_person_dict[key]) ###Adds to export list
            if p_persons_counter_dict[key] < self.OUT_OF_PERMANENCE_THRESHOLD: ###Person absent for too long
                p_person_dict.pop(key,None)
        ### part 3 - end ###
        return return_list