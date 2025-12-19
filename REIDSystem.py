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
from IDSystem import IDSystem
### Receives TEMP_PERSON list from IDSystem
### Uses DB to identify people, returning list of Person
### Also ADDs modification to DB on queue that receives from outside



"""
Just pass to the specified reid system -> or does nothing at all.
ReidSystem(list_of_temporary_person) will return list of modified TempPerson, that has the ID of the permanent Person
""" 
@dataclass
class REIDSystem:
    person_db : PersonDB = field(default_factory=PersonDB) ###
    reid_type : str = "dummy" ###IF no entry is given, uses dummy
    reid_in_use : Any = field(init=False, repr=False, default=None)
    
    def __post_init__(self):
        if reid_type == "dummy":
            print("dummy reid system chosen. The reid will work as a placeholder, but will not register anything")
        if reid_type == "CLIP":
            reid_in_use = CLIP(self.person_db)
        else :
            print("choosen reid system not found. Will change to dummy instead")
            reid_type = "dummy"
    
    def __call__(self, list_of_temporary_person):
        if reid_type != "dummy":
            return reid_in_use(list_of_temporary_person)
        else:
            return list_of_temporary_person



@dataclass
class REID_type:
    reid_type = "dummy"


@dataclass
class CLIP(REID_type):
    ###Static parameters
    SIMILARITY_THRESHOLD = 0.85
    
    ###External
    person_db : PersonDB
    
    ###INTERNAL PARAMETER
    tp_eq_dict : dict = field(default_factory=dict)
    
    def __post_init__(self):
        reid_type = "CLIP"
    
    
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
        ###Get features from all temporary persons in list
        for temp_person in list_of_temporary_person:
            temp_id = temp_person.id
            features_from_tp = yolo_to_pil(frame,temp_person) ###Gets features from temporary person
                
            if temp_id in self.tp_eq_dict: ###Equivalence is already in dict. JUST UPDATE THE FEATURES. DO NOT TRY TO OVERCORRECT THE PROGRAM.
                p_id = self.tp_eq_dict[temp_id] ### gets permanent person id from dict
                person = self.person_db.get_person_by_id(p_id) ### get person
                person.features = (person.features+features_from_tp)/2 ###Updates person feature
                
                ### Modifies Temporary person id
                temp_person.id = p_id
            
            ###Its not in dict. Must compare with people DB.
            else: 
                max_similarity = 0
                person_id = 0
                ###Check for all people in DB
                for real_person in self.person_db: 
                    similarity = compare(real_person.features,features_from_tp)
                    ###Find person in DB with bigger similarity(over threshold)
                    if (similarity > SIMILARITY_THRESHOLD) and (similarity > max_similarity):
                        max_similarity=similarity
                        person_id = real_person.id
                
                ###Compare with existing people. Match? Yes
                if max_similarity>0:
                    self.tp_eq_dict[temp_id]=person_id ###Keep in mind that two yolo temporary ids may share one person ID. However, it's better to acept that, and fix the similarity comparator, than to try to fix it in the atribution process
                    person = self.person_db.get_person_by_id(person_id) ### get person
                    person.features = (person.features+features_from_tp)/2 ###Updates person feature
                    
                    temp_person.id = person_id  ### Modifies Temporary person id to be equal to set Id from DB
                    
                ###Compare with existing people. Match? No
                else:
                    new_id = self.person_db.size()+1 ###The size will be equivalent to the number of objects in the db. So, the next id is this number plus one
                    new_person = Person(new_id,1,features_from_tp)
                    self.person_db.add(new_person)
                    self.tp_eq_dict[temp_id] = new_id
        
                    temp_person.id = person_id ### Modifies Temporary person id to be equal to ID of new person
                    
        return list_of_temporary_person ###returns the list, that has been modified
### Memory edits will be processed at the processManager, to avoid write_over_read











### Debug ###

Testing_mode = False ###Used to test the model detecting and tracking, but not exporting personal atributes.

print_box_output = True #TRUE or False


if __name__ == "__main__":
    ###VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    fps = 30
    width, height = 1920, 1080
    video_writer = cv2.VideoWriter(
        'output_test_video.mp4',
        fourcc,
        fps,
        (width, height)
    )

    id_system = IDSystem() ###create an ID system
    person_db = PersonDB()
    reid_system = REIDSystem(person_db)
    
    test_source = "auxiliares/People_in_line.mp4"
    cap = cv2.VideoCapture(test_source)

    # Verifica se abriu
    if not cap.isOpened():
        print("Erro ao abrir vídeo")
        exit()
    while True:
        ret, frame = cap.read()
        if Testing_mode == True:
             saida = id_system.testing(frame)
             tracker_result, temporary_persons = saida["result"], saida["temporary_persons"]
             reid_system_result = reid_system(temporary_persons)
        else:
            saida = id_system(frame)
            tracker_result, temporary_persons = saida["result"], saida["temporary_persons"]
            reid_system_result = reid_system(temporary_persons)
        
        if print_box_output==True:
            if temporary_persons:
                print("\n","||==========<>==========<>==========<>==========<>==========||","\n",
                      "first temp_person","\n",
                      temporary_persons[0],"\n",
                      "||==========<>==========<>==========<>==========<>==========||","\n")  ### a way to find exactly what is the output from the YOLO tracker
            print("boxes:","\n",
            "||----------|----------|----------|----------|----------|----------||","\n",
            tracker_result[0].boxes,"\n")
            
        
        if not ret:
            break  # End
        
        video_writer.write(tracker_result[0].plot())
        
    # 3. Release
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
        
