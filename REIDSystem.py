import cv2
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import threading
from queue import Queue
from typing import Any
### For using in CLIP
from PIL import Image
import clip

from TempPerson import TempPerson
from Person import Person
from PersonDB import PersonDB
### Receives TEMP_PERSON list from IDSystem
### Uses DB to identify people, returning list of Person
### Also ADDs modification to DB on queue that receives from outside

@dataclass
class REIDSystem: ###Just pass to the specified reid system -> or does nothing at all.
    reid_queue : Queue = field(default_factory=Queue) ###exports edits
    person_db : PersonDB = field(default_factory=PersonDB) ###
    reid_type : str = "dummy"
    reid_in_use : Any
    
    def __post_init__:
        if reid_type == "dummy":
            print("dummy reid system chosen. The reid will work as a placeholder, but will not register anything")
        if reid_type == "CLIP":
            reid_in_use = CLIP(person_db)
        else :
            print("choosen reid system not found. Will change to dummy instead")
            reid_type = "dummy"
    
    def __call__(self, list_of_temporary_person):
        if reid_type != "dummy":
            reid_in_use(list_of_temporary_person)
        else:
            return list_of_temporary_person


@dataclass
class CLIP:
    ###Static parameters
    SIMILARITY_THRESHOLD = 0.85
    
    
    ###
    person_db : PersonDB
    person_equivalent_dict : dict = field(default_factory=dict)
    
    def yolo_to_pil(frame,temp_person):
        ###For cutting image out of BBOX
        bbox = temp_person.bb
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        person_img = frame[y1:y2, x1:x2] 
        ###Fix image to be used
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(person_rgb)
        ###Process using clip
        processed = clip_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():  ###Using untrained model. Is faster.
            features = clip_model.encode_image(processed) 
        return features
    
    def compare(vector1, vector2):
        similarity = torch.cosine_similarity(vector1, vector2) ###cossine similarity
        
        return similarity.item()
        
    
    def __call__(self,frame,list_of_temporary_person):
        tp_eq_dict = {} ###dictionary that saves the relation of Identifier ID(in temp_person) to Person ID in DB : temporary_persons_equivalence_dictionary(tp_eq_dict)
        
        ###Get features from all temporary persons in list
        for temp_person in list_of_temporary_person:
                t_id = temp_person.id
                features_from_tp = yolo_to_pil(frame,temp_person) ###gets features
                
                """Compare with existing person in db. There is two paths here:
                1 - The person is already in reid dict, and you must compare with only 1 person.
                2 - It's not, and a new equivalence must be added, or a new person.
                """
                
                if t_id in tp_eq_dict: ###Equivalence is already in dict. JUST UPDATE THE FEATURES. DO NOT TRY TO OVERCORRECT THE PROGRAM.
                    p_id = tp_eq_dict[t_id] ### gets permanent person id from dict
                    person = self.person_db.get_person_by_id(p_id) ### get person
                    person.features = (person.features+features_from_tp)/2 ###Updates person feature
                else: ###Its not in dict. Must compare with people DB.
                
        
        ###Comparing
        
        
        
        ###Add to dict
        
        
        
        



### Memory edits will be processed at the processManager, to avoid write_over_read