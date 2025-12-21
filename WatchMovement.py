### Process 1, phase 2: find sync_movement
from dataclasses import dataclass, field
from typing import Any
from TempPerson import TempPerson
from scipy.stats import trim_mean
from ultralytics.utils.metrics import bbox_iou


@dataclass
class WatchMovement:
    permanent_persons_counter_dict : dict[int, int] = field(default_factory=dict) ###dict with id -> times seen
    changing_pos_dict              : dict[int, int] = field(default_factory=dict) ###dict with id -> value. When someone moves(IOU), this value starts increasing, till it gets to NEW_POS_THRESHOLD. Then, if the person is not moving, it updates their POS, and goes to -MOVING_THRESHOLD. It will increase till it gets to 0. When this happens, their movement is reset to 0.
    persons_position_dict          : dict[int,  bb] = field(default_factory=dict) ###dict id -> [frames_after_move,old_pos] ||> when gets to zero, update pos AND mov_dic
    persons_mov_dict               : dict[int,[float]] = field(default_factory=dict) ###dict id -> movement
    DISCART_THRESHOLD              : int   = -5
    SAME_PLACE_THRESHOLD           : int   = 0.6 ### IoU that defines someone that hasn't moved
    NEW_POS_THRESHOLD              : int   = 20
    MOVING_THRESHOLD               : int   = 10  ###Number of frames before forgeting the old position
    
    def __call__(list_from_WP : list[TempPerson]): ###list_of_temporary_persons_from_WatchPermanence
        p_pc_dict = self.permanent_persons_counter_dict
        p_pos_dict = self.persons_position_dict
        
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
        
        ### part 3 - checar quem se moveu ### -> atualizar 