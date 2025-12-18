from dataclasses import dataclass, field
from typing import Any
from ultralytics import YOLO
from TempPerson import TempPerson
### This class aims to manage the Id(identification) system. The temporal management bellongs to the ProcessManager, not this class. This class ONLY receives FRAMES(inputs) and gives Temp_Person as answers.


@dataclass
class IdSystem: ### Ideally, should be able to use more than one model
    model_type = str = "YoloID8n"
    model_in_use = Any
    
    def __post_init__(self): ###Creates the instance of the selected model
        if self.model_type == "YoloID8n":
            self.model_in_use = YoloID8n()
    
    def __call__(self,source):
        if self.model_type == "YoloID8n":
            model_analysis = self.model_in_use(source)
            return model_analysis
        else:
            print("Please, select an avaliable model, or use the standard, by giving no input")





###YOLO
@dataclass
class YoloID8n():
    # YOLO('yolov8s.pt')            # Small
    # YOLO('yolov8m.pt')            # Medium  
    # YOLO('yolov8l.pt')            # Large
    # YOLO('yolov8x.pt')            # XLarge (mais preciso, mais lento)
    model = YOLO('yolov8n.pt') ### You might use any of this
    
    def __call__(self,source):
        tracker_result = self.model.track(
        source=source,  #
        tracker='strongsort.yaml',  # or 'botsort.yaml'
        show=True,
        save=True,
        classes=[0],  # Filter classes: 0=person
        conf=0.5,           # Confidence threshold
        iou=0.5,            # IOU threshold
        device='cuda',      # 'cpu' or 'cuda' (GPU)
        verbose=True        # Show logs
        )
        
        temporary_persons = []
        for element in tracker_result[0]: ###The result from model.track is always a list, in which each position is respective to a frame. However, in our case, there's only 1 frame ever. So, just the first position is accounted for.
            t_person = TempPerson()
            t_person.id = element["track_id"]
            t_person.bb = element['bbox']
            t_person.confidence = element['confidence']
            t_person.position = element['position']
            temporary_persons.append(t_person)
        return temporary_persons
        



### Testing
if __name__ == "__main__":
    
    
        