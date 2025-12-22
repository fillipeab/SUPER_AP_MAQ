### Process 1, phase 2: find sync_movement
from dataclasses import dataclass, field
from typing import Any

from ultralytics.utils.metrics import bbox_iou
from sklearn.cluster import DBSCAN
import torch
import numpy as np

from TempPerson import TempPerson

@dataclass
class MovementWatcher:
    ### MovementWatcher[list[TempPerson]] -> list[TempPerson]
    permanent_people_counter_dict  : dict[int, int] = field(default_factory=dict) ###dict with id -> times seen
    people_dict                    : dict[int,  TempPerson] = field(default_factory=dict) ### dict wit id -> temporary_person
    changing_pos_dict              : dict[int, int] = field(default_factory=dict) ###dict with id -> value. When someone moves(IOU), this value starts increasing, till it gets to NEW_POS_THRESHOLD. Then, it updates their POS, and goes to -MOVING_THRESHOLD. It will increase till it gets to 0. When this happens, their movement is reset to 0.
    people_mov_dict                : dict[int, Any] = field(default_factory=dict) ###dict id -> movement
    SAME_PLACE_IOU                 : float   = 0.9 ### IoU that defines someone that has started a movement
    CYCLES_TO_UPDATE_POS           : int     = 240  ### Cycles between start of movement and "end". That is, to register the new position, and the movement
    CYCLES_TO_FORGET_MOVE          : int     = 480 ###Number of cycles before forgeting the old position and movement direction
    TIME_TO_FORGET                 : int     = 30  ###Frames before someone is erased from dicts
    iterator                       : int     = 0

    def __call__(self, list_from_WP : list[TempPerson]): ###list_of_temporary_people_from_PermanenceWatcher
        p_pc_dict = self.permanent_people_counter_dict
        people_dict = self.people_dict
        p_changing_pos_dict = self.changing_pos_dict
        p_mov_dict = self.people_mov_dict
        self.iterator+=1
        ### part 1 - entry interaction ###
        for temp_person in list_from_WP:
            if temp_person: ###avoids empty lines
                ### PC DICT UPDATING - 1.1 ###
                tp_id = temp_person.id
                if tp_id in p_pc_dict: ### ALREADY IN DICT
                    p_pc_dict[tp_id] = (self.TIME_TO_FORGET)
                    ### end ###    
                    ### MOVEMENT CHECKING ###
                    if p_changing_pos_dict[tp_id] <= 0: ### >0 means its already moving. =<0 means it either has been moved, or is still in place
                        ### print(temp_person.bb," compare ",people_dict[tp_id].bb)   
                        iou_matrix = round(float(bbox_iou(temp_person.bb, people_dict[tp_id].bb)),3) ###to check for movement
                        if iou_matrix < self.SAME_PLACE_IOU:
                            ### print("someone has started MOVING:",tp_id,"turn: ",self.iterator)
                            p_changing_pos_dict[tp_id]=1  ### starts the counting
                            
                    else: ###It is under change
                        if p_changing_pos_dict[tp_id]==self.CYCLES_TO_UPDATE_POS: ###has reach the threshold. Changes must be accounted for
                            p_mov_dict[tp_id]=temp_person.bb-people_dict[tp_id].bb ###updates movement
                            people_dict[tp_id]=temp_person ###updates person
                            p_changing_pos_dict[tp_id] = -self.CYCLES_TO_FORGET_MOVE
                            
                            
                ### END OF UPDATING ###
                else: ###NOT IN DICT
                ### Added to dict now
                    p_pc_dict[tp_id]  = self.TIME_TO_FORGET
                    people_dict[tp_id] = temp_person
                    p_changing_pos_dict[tp_id] = 0
                    p_mov_dict[tp_id] = torch.tensor([0,0,0,0])
                ### FINDING THE MOVEMENT THRESHOLD ###
        ### part 1 - end ###
        
        ### part 2 - update memory
        remove_from_p_pc_dict = []
        for key in p_pc_dict:
            p_pc_dict[key]-=1
            if p_pc_dict[key] <= 0: ###Remove from memory person absent for too long - avoids memory overflow
                remove_from_p_pc_dict.append(key)
        for key in remove_from_p_pc_dict: ##remove from dict##
            people_dict.pop(key,None)
            p_pc_dict.pop(key,None)
            p_changing_pos_dict.pop(key,None)
            p_mov_dict.pop(key,None)
        ### part 2 - end ###
        
        ### part 3 - counting ###
        for key in p_changing_pos_dict:
            if p_changing_pos_dict[key] > 0:
                if p_changing_pos_dict[key] < self.CYCLES_TO_UPDATE_POS:
                    p_changing_pos_dict[key]+=1
                else: ###means threshold was reached
                    pass
            elif p_changing_pos_dict[key] < 0: ###zeros the movement registry after n cicles
                p_changing_pos_dict[key]+=1
                if p_changing_pos_dict==0:
                    p_mov_dict[key]=torch.tensor([0,0,0,0])
        
        ### PART 4 - finally, time to export people in sync movement ###
        moved_people_dict = {k: v for k, v in p_mov_dict.items() if torch.any(v != 0)} ### creates a new dict, where it's only the people that have moved in some cordinate
        
        ### Part 5 - Checks for the biggest cluster of tensors -> which will correspond to the direction of the line
        ### IMPORTANT: ITS CONSIDERED THAT SYNCRONIZED MOVEMENT, IN THE AREA OF THE CAMERA, IS THE MAIN LINE. REMEMBER, THAT IS PEOPLE CONSISTENTLY IN IMAGE GOING IN THE SAME DIRECTION
        selected_group = find_movement_group(moved_people_dict) ###Will only return something if the group is big enough
        ### Make the exporting list ###
        list_of_people_in_sync_movement = []
        if len(selected_group) > 0:
            ###print("finally, A GROUP!")
            for key in selected_group:
                list_of_people_in_sync_movement.append(people_dict[key]) ###return list of temporary people, with bb
        
        return list_of_people_in_sync_movement
        
        
        

###function to find the bigest group - will be used to find syncronous movement

def find_movement_group(dict_tensors, dir_weight=4, mag_weight=1, threshold=0.6):
    n = len(dict_tensors)
    if n < 2:
        return list(dict_tensors.keys())
    
    keys = list(dict_tensors.keys())
    features = torch.stack(list(dict_tensors.values())).cpu().numpy()
    
    mags = np.linalg.norm(features, axis=1)
    moving = mags > 0.1
    
    if np.sum(moving) < 2:
        return keys
    
    dirs = np.zeros_like(features)
    valid_mags = mags[moving]
    valid_features = features[moving]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dirs[moving] = valid_features / valid_mags[:, np.newaxis]
    dirs = np.nan_to_num(dirs)
    
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(dirs[moving])
    labels = clustering.labels_
    
    if np.all(labels == -1):
        return keys
    
    main_label = max(set(labels) - {-1}, key=list(labels).count)
    moving_indices = np.where(moving)[0]
    main_mask = labels == main_label
    main_indices = moving_indices[main_mask]
    
    if len(main_indices) < 2:
        return [keys[i] for i in main_indices] if len(main_indices) > 0 else keys
    
    group_features = features[main_indices]
    group_mags = mags[main_indices]
    avg_mag = np.mean(group_mags)
    
    if avg_mag < 0.01:
        return [keys[i] for i in main_indices]
    
    centroid = np.mean(dirs[main_indices], axis=0)
    final_members = []
    
    for idx in main_indices:
        person_dir = features[idx] / max(mags[idx], 0.01)
        dir_sim = max(0, 1.0 - 0.5 * np.linalg.norm(person_dir - centroid))
        
        mag_diff = abs(mags[idx] - avg_mag) / avg_mag
        mag_sim = max(0, 1.0 - min(mag_diff, 1.0))
        
        score = (dir_weight * dir_sim + mag_weight * mag_sim)/(dir_weight+mag_weight)
        if score >= threshold:
            final_members.append(idx)
    
    return [keys[i] for i in final_members] if len(final_members) >= 2 else [keys[i] for i in main_indices]


### OLD VERSION ####

def find_cluster(dict_tensors, eps=0.5, min_samples=2):
    """Retorna chaves do maior agrupamento de tensores similares."""
    if not dict_tensors:
        return [], 0
    
    keys = list(dict_tensors.keys())
    features = torch.stack(list(dict_tensors.values())).cpu().numpy()
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    
    positive_labels = labels[labels >= 0]
    if len(positive_labels) == 0:
        return [], 0
    
    # Conta ocorrências de cada label positivo
    unique_labels, contagens = np.unique(positive_labels, return_counts=True)
    selected_label = unique_labels[np.argmax(contagens)]
    
    # Pega chaves do maior cluster
    selected_cluster = [keys[i] for i, lbl in enumerate(labels) if lbl == selected_label]
    
    return selected_cluster, len(selected_cluster)


### OLD VERSION ###  