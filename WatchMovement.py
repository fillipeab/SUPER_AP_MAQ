### Process 1, phase 2: find sync_movement
from dataclasses import dataclass, field
from typing import Any
from TempPerson import TempPerson
from scipy.stats import trim_mean


@dataclass
class WatchMovement:
    permanent_persons_counter_dict : dict[int,int] = field(default_factory=dict) ###dict with id -> times seen
    persons_position_dict          : dict[int,[float,int] = field(default_factory=dict) ###dict que faz id -> [frames sem se mover, pos]
    movement_threshold             : int   = 0 ###Ideally, its best to define it as 1,5 times the smaller value of a person square[X or Y]. This value is likely to be their widht. The best is to consider an average.
    MOVEMENT_THRESHOLD_COUNT       : int   = 0
    MOVEMENT_THRESHOLD_UPDATE      : int   = 100
    DISCART_THRESHOLD              : int   = -5
    OLD_POS_THRESHOLD              : int   = 20 ###Number of frames "in same place" to forget the old movement
    
    def __call__(list_from_WP : list[TempPerson]): ###list_of_temporary_persons_from_WatchPermanence
        p_pc_dict = self.permanent_persons_counter_dict
        p_pos_dict = self.persons_position_dict
        
        
        ### part 0 ###
        ###Reconfigurate movement threshold###
        if self.MOVEMENT_THRESHOLD_COUNT>=self.MOVEMENT_THRESHOLD_UPDATE:
            self.movement_threshold=0
            self.MOVEMENT_THRESHOLD_COUNT=0
        self.MOVEMENT_THRESHOLD_COUNT+=1
        ###finding movement threshold
        if self.movement_threshold == 0:
           smaller_side = []
           for temp_person in list_from_WP:
                x1, y1, x2, y2 = temp_person.bb
                smaller = (x2 - x1)
                if (y2-y1)<(smaller):
                    smaller = (y2-y1)
                smaller_side.append(smaller)   
            self.movement_threshold=trim_mean(smaller_side,0.2)  ###makes the mean, excluding the 20% mais extremos de cada lado
        ### part 0 - end ###
        
        ### part 1 - entry interaction ###
        for temp_person in list_from_WP:
            
            ### PC DICT UPDATING ###
            tp_id = temp_person.id
            if tp_id in p_pc_dict: 
            ### person already in dict
            
                if p_pc_dict[tp_id]<0:  ###Deals with Discart_threshold###
                    p_pc_dict[tp_id]=0
                p_pc_dict[tp_id] += 2 ###important
                ### end ###
            ###not in dict - add to id ###    
            
            else:
            ### Added to dict now
                p_pc_dict[tp_id]  = 2
                p_pos_dict[tp_id] = [0, temp_person.bb]
            ### FINDING THE MOVEMENT THRESHOLD ###
                
        ### part 1 - end ###
        
        ### part 2 - update memory
        remove_from_p_pc_dict = []
        for key in p_pc_dict:
            p_pc_dict[key]-=1
            ### Maximum limit
            if p_pc_dict[key] > self.LIMIT_SEEN:
                p_pc_dict[key] = LIMIT_SEEN
            ### Minimum limit
            if p_pc_dict[key] < -self.DISCART_THRESHOLD: ###Remove from memory person absent for too long - avoids memory overflow
                remove_from_p_pc_dict.append(key)
        ##remove from dict##
        for key in remove_from_p_pc_dict:
            p_pos_dict.pop(key,None)
            p_pc_dict.pop(key,None)
        ### part 2 - end ###
        