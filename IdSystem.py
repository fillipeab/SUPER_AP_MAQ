import cv2
import torch
import numpy as np
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
        return tracker_result
        """
        temporary_persons = []
        for frame in tracker_result: ### By definition, there should be ONLY one frame. However, in the possibility of having more than one, this implementation was done. It's important to note that it will leave ALL detections in the same list. That is, there will be no frame differentiation.
            for element in frame.boxes:
                ### FIX THIS AND IMPROVE ####
                t_person = TempPerson()
                t_person.id = element["track_id"]
                t_person.bb = element['bbox']
                t_person.confidence = element['confidence']
                t_person.position = element['position']
                temporary_persons.append(t_person)
        return temporary_persons
        """



### Testing
if __name__ == "__main__":
    id_system = IdSystem()
    test_source = "auxiliares/People_in_line.mp4"
    detections = []
    cap = cv2.VideoCapture(test_source)

    # Verifica se abriu
    if not cap.isOpened():
        print("Erro ao abrir vídeo")
        exit()
    while True:
        ret, frame = cap.read()
        detection = id_system(frame) ### checking detection
        print(detection,"\n")
        if not ret:
            break  # End
        # show frame
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 3. Release recursos
    cap.release()
    cv2.destroyAllWindows()
    
        