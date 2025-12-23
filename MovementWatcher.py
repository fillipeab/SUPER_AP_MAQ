### Process 1, phase 2: find sync_movement
from dataclasses import dataclass, field
from typing import Any

from ultralytics.utils.metrics import bbox_iou
from sklearn.cluster import DBSCAN
import torch
from copy import deepcopy
import numpy as np

from TempPerson import TempPerson

@dataclass
class MovementWatcher:
    ### MovementWatcher[list[TempPerson]] -> list[TempPerson]
    permanent_people_counter_dict  : dict[int, int] = field(default_factory=dict) ###dict with id -> times seen
    people_dict                    : dict[int,  TempPerson] = field(default_factory=dict) ### dict wit id -> temporary_person
    changing_pos_dict              : dict[int, int] = field(default_factory=dict) ###dict with id -> value. When someone moves(IOU), this value starts increasing, till it gets to NEW_POS_THRESHOLD. Then, it updates their POS, and goes to -MOVING_THRESHOLD. It will increase till it gets to 0. When this happens, their movement is reset to 0.
    people_mov_dict                : dict[int, Any] = field(default_factory=dict) ###dict id -> movement
    SAME_PLACE_IOU                 : float   = 0.95 ### IoU that defines someone that has started a movement
    CYCLES_TO_UPDATE_POS           : int     = 12  ### Cycles between start of movement and "end". That is, to register the new position, and the movement
    CYCLES_TO_FORGET_MOVE          : int     = 72 ###Number of cycles before forgeting the old movement direction
    TIME_TO_FORGET                 : int     = 48+CYCLES_TO_UPDATE_POS  ###Frames before someone is erased from dicts
    iterator                       : int     = 0
    eps                            : float   = 1 ###really permissive

    def __call__(self, list_from_WP : list[TempPerson]): ###list_of_temporary_people_from_PermanenceWatcher
        p_pc_dict = self.permanent_people_counter_dict
        people_dict = self.people_dict
        p_changing_pos_dict = self.changing_pos_dict
        p_mov_dict = self.people_mov_dict
        self.iterator+=1
        ### part 1 - entry interaction ###
        ###print("here we go")
        for temp_person in list_from_WP:
            if temp_person: ###avoids empty lines
                ### PC DICT UPDATING - 1.1 ###
                tp_id = int(temp_person.id)
                if tp_id in p_pc_dict: ### ALREADY IN DICT
                    ###print("in dict")
                    p_pc_dict[tp_id] = (self.TIME_TO_FORGET)
                    ### end ###    
                    ### MOVEMENT CHECKING ###
                    if p_changing_pos_dict[tp_id] <= 0: ### >0 means its already moving. =<0 means it either has been moved, or is still in place
                        ### print(temp_person.bb," compare ",people_dict[tp_id].bb)   
                        iou_matrix = round(float(bbox_iou(temp_person.bb, people_dict[tp_id].bb)),3) ###to check for movement
                        ###print("maybe change? ",iou_matrix)
                        ###print(tp_id,temp_person.bb,people_dict[tp_id].bb)
                        if iou_matrix < self.SAME_PLACE_IOU:
                            ###print("someone has start MOVING: ",tp_id," turn: ",self.iterator," iou ",iou_matrix)
                            p_changing_pos_dict[tp_id]=1  ### starts the counting
                            
                    else: ###It is under change
                        if p_changing_pos_dict[tp_id]>=self.CYCLES_TO_UPDATE_POS: ###has reach the threshold. Changes must be accounted for ### EDIT 1 : Avoiding run erros
                            p_mov_dict[tp_id]=p_mov_dict[tp_id]+(temp_person.bb-people_dict[tp_id].bb) ####REAL CHANGE - WHAT IF MOVEMENT IS THE SUM, AND ONLY FORGET WHEN DONE
                            print(" turn: ",self.iterator,"person id: ",tp_id,"movement: ",p_mov_dict[tp_id])
                            people_dict[tp_id]=deepcopy(temp_person) ###updates person
                            p_changing_pos_dict[tp_id] = -self.CYCLES_TO_FORGET_MOVE
                            
                            
                ### END OF UPDATING ###
                else: ###NOT IN DICT
                ### Added to dict now
                    ###print("new person",tp_id)
                    p_pc_dict[tp_id]  = self.TIME_TO_FORGET
                    people_dict[tp_id] = deepcopy(temp_person)
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
                else: ###means threshold was reached - WILL ONLY update when they get back in frame
                    pass
            elif p_changing_pos_dict[key] < 0: ###zeros the movement registry after n cicles
                p_changing_pos_dict[key]+=1
                if p_changing_pos_dict[key]==0:
                    p_mov_dict[key]=torch.tensor([0,0,0,0])
        ### part 3 - counting - end ###

        ### PART 4 - finally, time to export people in sync movement ###
        moved_people_dict = {}
        set_of_moving_people = set()
        for key in p_mov_dict:
            if torch.any(p_mov_dict[key]!= 0):
                movement = p_mov_dict[key]
                print(movement)
                moved_people_dict[key] = movement
                set_of_moving_people.add(key)
        
        ### Part 5 - Checks for the biggest cluster of tensors -> which will correspond to the direction of the line
        ### IMPORTANT: ITS CONSIDERED THAT SYNCRONIZED MOVEMENT, IN THE AREA OF THE CAMERA, IS THE MAIN LINE. REMEMBER, THAT IS PEOPLE CONSISTENTLY IN IMAGE GOING IN THE SAME DIRECTION
        selected_group = []
        if moved_people_dict:
            selected_group, _ = find_movement_group_v2(moved_people_dict, eps = self.eps) ###Will only return something if the group is big enough
        ### Make the exporting list ###
        list_of_people_in_sync_movement = []
        if len(selected_group) > 0:
            ###print("finally, A GROUP!")
            for key in selected_group:
                list_of_people_in_sync_movement.append(people_dict[key]) ###return list of temporary people, with bb
        
        return {"sync_moving" : list_of_people_in_sync_movement, "set_of_moving_people" : set_of_moving_people}
        
        
        

###function to find the bigest group - will be used to find syncronous movement

def find_movement_group_v2(dict_tensors, mag_weight=1, dir_weight=4, threshold=0.4, eps=0.5):
    if len(dict_tensors) < 2:
        return list(dict_tensors.keys()), len(dict_tensors)
    
    keys = list(dict_tensors.keys())
    X = torch.stack(list(dict_tensors.values())).cpu().numpy()
    
    mags = np.linalg.norm(X, axis=1)
    moving = mags > 0.1
    
    if moving.sum() < 2:
        return keys, len(keys)
    
    D = np.zeros_like(X)
    D[moving] = X[moving] / mags[moving, None]
    D = np.nan_to_num(D)
    
    labels = DBSCAN(eps=eps, min_samples=2).fit(D[moving]).labels_
    
    if (labels >= 0).sum() == 0:
        ###print("No positive")
        return keys, len(keys)
    
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    main_label = unique[counts.argmax()]
    
    moving_idx = np.where(moving)[0]
    main_idx = moving_idx[labels == main_label]
    
    if len(main_idx) < 2:
        ###print("few positive")
        selected = [keys[i] for i in main_idx] if main_idx.size > 0 else keys
        return selected, len(selected)
    
    main_mags = mags[main_idx]
    avg_mag = main_mags.mean()
    
    if avg_mag < 0.01:
        return [keys[i] for i in main_idx], len(main_idx)
    ###print("finally")
    centroid = D[main_idx].mean(axis=0)
    final_idx = []
    
    for idx in main_idx:
        dir_dist = np.linalg.norm(D[idx] - centroid)
        dir_sim = max(0, 1.0 - 0.5 * dir_dist)
        
        mag_diff = abs(mags[idx] - avg_mag) / avg_mag
        mag_sim = max(0, 1.0 - min(mag_diff, 1.0))
        
        score = (dir_weight * dir_sim + mag_weight * mag_sim) / (dir_weight + mag_weight)
        if score >= threshold:
            final_idx.append(idx)
    
    result = [keys[i] for i in final_idx] if len(final_idx) >= 2 else [keys[i] for i in main_idx]
    ###print("do we have a result????", len(result))
    return result, len(result)
### OLD VERSION ####


def find_movement_group(dict_tensors, dir_weight=4, mag_weight=1, threshold=0.4, eps = 0.5): ###EPS defines how flexible it is
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
    
    clustering = DBSCAN(eps=eps, min_samples=2).fit(dirs[moving])
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
        person_dir = dirs[idx]
        dir_sim = max(0, 1.0 - 0.5 * np.linalg.norm(person_dir - centroid))
        
        mag_diff = abs(mags[idx] - avg_mag) / avg_mag
        mag_sim = max(0, 1.0 - min(mag_diff, 1.0))
        
        score = (dir_weight * dir_sim + mag_weight * mag_sim)/(dir_weight+mag_weight)
        if score >= threshold:
            final_members.append(idx)
    return_cluster = [keys[i] for i in final_members] if len(final_members) >= 2 else [keys[i] for i in main_indices]
    return return_cluster , len(return_cluster)





def find_cluster(dict_tensors, eps=0.5, min_samples=2):
    """Retorna chaves do maior agrupamento de tensores similares."""
    ###print("find_cluster was called")
    if not dict_tensors:
        
        print("and had nothing to return")
        return [], 0
    
    keys = list(dict_tensors.keys())
    features = torch.stack(list(dict_tensors.values())).cpu().numpy()
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    
    positive_labels = labels[labels >= 0]
    if len(positive_labels) == 0:
        print("No positive labels")
        return [], 0
    print("labels found")
    # Conta ocorrências de cada label positivo
    unique_labels, contagens = np.unique(positive_labels, return_counts=True)
    selected_label = unique_labels[np.argmax(contagens)]
    
    # Pega chaves do maior cluster
    selected_cluster = [keys[i] for i, lbl in enumerate(labels) if lbl == selected_label]
    
    print(len(selected_cluster))
    return selected_cluster, len(selected_cluster)



def find_cluster_v2(dict_tensors, eps=0.5, min_samples=2): ###USING NORMS
    if not dict_tensors:
        return [], 0
    
    keys = list(dict_tensors.keys())
    features = torch.stack(list(dict_tensors.values())).cpu().numpy()
    
    # Normalização pela norma
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-8)
    
    # DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_norm)
    labels = clustering.labels_
    
    # Maior cluster
    positive = labels[labels >= 0]
    if len(positive) == 0:
        print("No possitive labels")
        return [], 0
    
    unique, counts = np.unique(positive, return_counts=True)
    main_label = unique[np.argmax(counts)]
    
    cluster_keys = [keys[i] for i, lbl in enumerate(labels) if lbl == main_label]
    return cluster_keys, len(cluster_keys)
### OLD VERSION ###  