### Process 1, phase 2: find sync_movement
from dataclasses import dataclass, field
from typing import Any
from TempPerson import TempPerson
from scipy.stats import trim_mean
from ultralytics.utils.metrics import bbox_iou
import torch


@dataclass
class WatchMovement:
    permanent_persons_counter_dict : dict[int, int] = field(default_factory=dict) ###dict with id -> times seen
    persons_dict                   : dict[int,  TempPerson] = field(default_factory=dict)
    changing_pos_dict              : dict[int, int] = field(default_factory=dict) ###dict with id -> value. When someone moves(IOU), this value starts increasing, till it gets to NEW_POS_THRESHOLD. Then, it updates their POS, and goes to -MOVING_THRESHOLD. It will increase till it gets to 0. When this happens, their movement is reset to 0.
    persons_mov_dict               : dict[int, tensor(4,)] = field(default_factory=dict) ###dict id -> movement
    DISCART_THRESHOLD              : int   = -5
    SAME_PLACE_IOU                 : int   = 0.1 ### IoU that defines someone that hasn't moved
    CYCLES_TO_UPDATE_POS           : int   = 20
    CYCLES_TO_FORGET_MOVE          : int   = 100 ###Number of frames before forgeting the old position and movement direction
    LIMIT_SEEN_COUNTER             : int   = 10

    def __call__(self, list_from_WP : list[TempPerson]): ###list_of_temporary_persons_from_WatchPermanence
        p_pc_dict = self.permanent_persons_counter_dict
        persons_dict = self.persons_dict
        p_changing_pos_dict = self.changing_pos_dict
        p_mov_dict = self.persons_mov_dict
        
        ### part 1 - entry interaction ###
        for temp_person in list_from_WP:
            
            ### PC DICT UPDATING - 1.1 ###
            tp_id = temp_person.id
            if tp_id in p_pc_dict: 
                ### person already in dict
                if p_pc_dict[tp_id]<0:  ###Deals with Discart_threshold###
                    p_pc_dict[tp_id]=0
                p_pc_dict[tp_id] += 2 ###important
                ### end ###    
                ### MOVEMENT CHECKING ###
                iou_matrix = bbox_iou(temp_person.bb, persons_dict[tp_id].bb) ###to check for movement
                
                if p_changing_pos_dict[tp_id]<=0: ### >0 means its already moving. =<0 means it either has been moved, or is still in place
                    if iou_matrix < self.SAME_PLACE_IOU:
                        p_changing_pos_dict[tp_id]=1  ### starts the counting
                else: ###It is under change
                    if p_changing_pos_dict[tp_id]==self.CYCLES_TO_UPDATE_POS: ###has reach the threshold. Changes must be accounted for
                        p_mov_dict[tp_id]=temp_person.bb-persons_dict[tp_id].bb ###updates movement
                        persons_dict[tp_id]=temp_person ###updates person
                        p_changing_pos_dict[tp_id] = -self.CYCLES_TO_FORGET_MOVE
                        
                        
            ### END OF UPDATING ###
            else:
            ### Added to dict now
                p_pc_dict[tp_id]  = 2
                persons_dict[tp_id] = temp_person
                p_changing_pos_dict[tp_id] = 0
                p_mov_dict[tp_id] = torch.tensor([0,0,0,0])
            ### FINDING THE MOVEMENT THRESHOLD ###
        ### part 1 - end ###
        
        ### part 2 - update memory
        remove_from_p_pc_dict = []
        for key in p_pc_dict:
            p_pc_dict[key]-=1
            ### Maximum limit
            if p_pc_dict[key] > self.LIMIT_SEEN_COUNTER:
                p_pc_dict[key] = LIMIT_SEEN_COUNTER
            ### Minimum limit
            if p_pc_dict[key] < -self.DISCART_THRESHOLD: ###Remove from memory person absent for too long - avoids memory overflow
                remove_from_p_pc_dict.append(key)
        ##remove from dict##
        for key in remove_from_p_pc_dict:
            persons_dict.pop(key,None)
            p_pc_dict.pop(key,None)
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
        moved_persons_dict = {}
        for key in p_mov_dict:
            if p_mov_dict[key]!=torch.tensor([0,0,0,0]):
                moved_persons_dict[key]=p_mov_dict[key] ###Adds to the moved_person_dict
        
        ### Part 5 - Checks for the biggest cluster of tensors -> which will correspond to the direction of the line
        ### IMPORTANT: ITS CONSIDERED THAT SYNCRONIZED MOVEMENT, IN THE AREA OF THE CAMERA, IS THE MAIN LINE. REMEMBER, THAT IS PEOPLE CONSISTENTLY IN IMAGE GOING IN THE SAME DIRECTION
        
        selected_cluster, cluster_len = find_cluster(moved_persons_dict) ###Will only return something if the group is big enough
        
        ### Make the exporting list ###
        list_of_persons_in_sync_movement = []
        if cluster_len > 0:
            for key in selected_cluster:
                list_of_persons_in_sync_movement.append(persons_dict[key]) ###return list of temporary persons, with bb
        
        return list_of_persons_in_sync_movement
        
        
        

###function to find the bigest cluster - will be used to find syncronous movement
from sklearn.cluster import DBSCAN
import torch
import numpy as np

def find_cluster(dict_tensors, eps=0.5, min_samples=2):
    """Retorna chaves do maior agrupamento de tensores similares."""
    if not dict_tensors:
        return [], 0
    
    keys = list(dict_tensores.keys())
    features = torch.stack(list(dict_tensores.values())).cpu().numpy()
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    
    positive_labels = labels[labels >= 0]
    if len(positive_labels) == 0:
        return [], 0
    
    # Conta ocorrências de cada label positivo
    unique_labels, contagens = np.unique(positive_labels, return_counts=True)
    selected_label = unique_labels[np.argmax(contagens)]
    
    # Pega chaves do maior cluster
    selected_cluster = [chaves[i] for i, lbl in enumerate(labels) if lbl == selected_label]
    
    return selected_cluster, len(selected_cluster)