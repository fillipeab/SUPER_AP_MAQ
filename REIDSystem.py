import cv2
import torch
from dataclasses import dataclass, field
from typing import Any, ClassVar
from typing import Any
### For using in CLIP
from PIL import Image
import mobileclip

from TempPerson import TempPerson
from Person import Person
from PersonDB import PersonDB
from IDSystem import IDSystem
### Receives TEMP_PERSON list from IDSystem
### Uses DB to identify people, returning list of Person
### Also ADDs modification to DB on queue that receives from outside


"""
Expected behavior
Entry : frame : frame, list_of_temporary_person : list[temporary_person]
Call : REIDSystem(frame,list_of_temporary_person)
Output : list_of_temporary_person(modified) : list[temporary_person]
"""


"""
Just pass to the specified reid system -> or does nothing at all.
ReidSystem(list_of_temporary_person) will return list of modified TempPerson, that has the ID of the permanent Person
""" 
@dataclass
class REIDSystem:
    person_db : PersonDB = field(default_factory=PersonDB) ###
    reid_type : str = "dummy" ###IF no entry is given, uses dummy
    __reid_in_use : Any = None
    WARN_ID_CHANGE : bool = False
    
    @property
    def reid_in_use(self):
        return self.__reid_in_use
    @reid_in_use.setter
    def reid_in_use(self,value):
        if not self.__reid_in_use: ###has not been defined yet || works so the reid_in_use is defined only one, for each REID system
            self.__reid_in_use=value
        else:
            pass
    
    
    def __post_init__(self):
        if self.reid_type == "mobileCLIP":
            self.reid_in_use = mobileCLIP(WARN_ID_CHANGE = self.WARN_ID_CHANGE, person_db = self.person_db)
        elif self.reid_type == "dummy":
            print("dummy reid system chosen. The reid will work as a placeholder, but will not register anything")
        else :
            print("choosen reid system not found. Will change to dummy instead")
            self.reid_type = "dummy"
    
    def __call__(self, frame, list_of_temporary_person):
        ###other
        if self.reid_type != "dummy":
            result = self.reid_in_use(frame, list_of_temporary_person)
            return result
        ###dummy
        else:
            return list_of_temporary_person



@dataclass
class REID_type: ###It's not used directly, but works as parent class to any future REID_type
    ###External
    person_db : PersonDB
    ###INTERNAL PARAMETER
    tp_eq_dict : dict = field(default_factory=dict)
    ### Show when there is an ID change
    WARN_ID_CHANGE : bool = True;
    
    ###Identifier of REID_type
    reid_type :str = "dummy"
    
    def change_id(self,temp_person,new_id,extra_print=""):
        if self.WARN_ID_CHANGE == True:
            if temp_person.id != new_id:
                print(extra_print)
                print("Id has been changed from ",temp_person.id," to ",new_id,"\n")
        temp_person.id=new_id
            

@dataclass
class mobileCLIP(REID_type):
    model : ClassVar[Any] = None
    preprocess : ClassVar[Any] = None
    SIMILARITY_THRESHOLD: ClassVar[float] = 0.85
    
    def __post_init__(self):
        self.reid_type = "mobileCLIP"
        if self.model is None:
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='modelo/mobileclip_s0.pt')
            self.model.eval()
    
    def yolo_to_pil(self,frame,temp_person):
        ###For cutting image out of BBOX
        bbox = temp_person.bb.int().tolist()
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        person_img = frame[y1:y2, x1:x2] 
        ###Fix image to be used
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(person_rgb)
        ###Process using clip
        processed = self.preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():  ###Using untrained model. Is faster.
            features = self.model.encode_image(processed) 
        return features
    
    def compare(self,vector1, vector2):
        similarity = torch.cosine_similarity(vector1, vector2) ###cossine similarity
        
        return similarity.item()
        
    
    def __call__(self,frame,list_of_temporary_person):
        ###Get features from all temporary people in list
        ###Keep in mind this list will NOT be preserved. Its very nature is being temporary
        for temp_person in list_of_temporary_person:
            temp_id = temp_person.id
            features_from_tp = self.yolo_to_pil(frame,temp_person) ###Gets features from temporary person
            

            ### Is there any entry in dict that corresponds to this ID(that was atributed by yolo)?
            if temp_id in self.tp_eq_dict: ###Equivalence is already in dict. JUST UPDATE THE FEATURES. DO NOT TRY TO OVERCORRECT THE PROGRAM.
                p_id = self.tp_eq_dict[temp_id] ### gets permanent person id from dict
                person = self.person_db.get_person_by_id(p_id) ### get person
                person.features = (person.features+features_from_tp)/2 ###Updates person feature
                
                ### Modifies Temporary_person id in list to match
                self.change_id(temp_person,p_id,"already in dict\n")
            
            ###Its not in dict. Must compare with people DB.
            else: 
                max_similarity = 0
                person_id = 0
                ###Check for all people in DB
                for i in range(self.person_db.size): 
                    real_person = self.person_db.get_person_by_id(i)
                    if real_person: ###Just checking
                        similarity = self.compare(real_person.features,features_from_tp)
                        ###Find person in DB with bigger similarity(over threshold)
                        if (similarity > self.SIMILARITY_THRESHOLD) and (similarity > max_similarity):
                            max_similarity=similarity
                            person_id = real_person.id
                
                ###"Compare with existing people. Match?" Yes
                if max_similarity>0:
                    self.tp_eq_dict[temp_id]=person_id ###Add key
                    ###Keep in mind that two yolo temporary ids may share one person ID. However, it's better to acept that, and fix the similarity comparator, than to try to fix it in the atribution process.
                    person = self.person_db.get_person_by_id(person_id) ### get person
                    person.features = (person.features+features_from_tp)/2 ###Updates person feature
                    
                    self.change_id(temp_person,person_id,"match with person in db\n") ### Modifies Temporary person id to be equal to set Id from DB
                    
                ###"Compare with existing people. Match?" No
                else:
                    new_id = self.person_db.size ###The size is equivalent to the number of objects in the DB. However, as id counting starts in 0, so this number is also the id of the next element to be added.
                    new_person = Person(new_id,features_from_tp)
                    self.person_db.add(new_person)
                    self.tp_eq_dict[temp_id] = new_id
                    
                    ### Modifies Temporary person id to be equal to ID of new person
                    a = "new entry in dict\nnew person id :{}".format( new_person.id)
                    self.change_id(temp_person,new_id,a)
                    
        return list_of_temporary_person ###returns the list, that has been modified
### Memory edits will be processed at the processManager, to avoid write_over_read











### Debug ###

Testing_mode = False ###Used to test the model detecting and tracking, but not exporting personal atributes.

print_box_output = False #TRUE or False


if __name__ == "__main__":
    print("REID testing")
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
    reid_system = REIDSystem(person_db,"mobileCLIP")
    
    test_source = "auxiliares/People_in_line.mp4"
    cap = cv2.VideoCapture(test_source)

    # Verifica se abriu
    if not cap.isOpened():
        print("Erro ao abrir vídeo")
        exit()
    while True:
        ret, frame = cap.read()
        if Testing_mode == True:
             model_analysis = id_system.testing(frame)
             result, temporary_people = model_analysis["result"], model_analysis["temporary_people"]
             reid_system_result = reid_system(frame,temporary_people)
        else:
            model_analysis = id_system(frame)
            result, temporary_people = model_analysis["result"], model_analysis["temporary_people"]
            reid_system_result = reid_system(frame,temporary_people)
        
        if print_box_output==True:
            if temporary_people:
                print("\n","||==========<>==========<>==========<>==========<>==========||","\n",
                      "first temp_person","\n",
                      temporary_people[0],"\n",
                      "||==========<>==========<>==========<>==========<>==========||","\n")  ### a way to find exactly what is the output from the YOLO tracker
            print("boxes:","\n",
            "||----------|----------|----------|----------|----------|----------||","\n",
            result[0].boxes,"\n") ### !!!!!!!!!!!!!!!!!!!!!!! This line is sensitive to the model type !!!!!!!!!!!!!!!!!!!!!!!!!!!
            
        
        if not ret:
            break  # End
        
        video_writer.write(result[0].plot()) ### !!!!!!!!!!!!!!!!!!!!!!! This line is sensitive to the model type !!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    # 3. Release
    print("||====================||END||====================||")
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
        
