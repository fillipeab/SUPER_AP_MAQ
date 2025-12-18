import cv2
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from ultralytics import YOLO
from TempPerson import TempPerson
### This class aims to manage the Id(identification) system. The temporal management bellongs to the ProcessManager, not this class. This class ONLY receives FRAMES(inputs) and gives Temp_Person(s) as answers.


@dataclass
class IdSystem: ### Ideally, should be able to use more than one model
    model_type = str = "YoloID8n"
    model_in_use = Any
    
    def __post_init__(self): ###Creates the instance of the selected model
        if self.model_type == "YoloID8n": 
            self.model_in_use = YoloID8n()

    def __call__(self,source):
        if self.model_type == "YoloID8n": ###Making sure the model does exist
            model_analysis = self.model_in_use(source) ###The model was already created in post_init. New models should, by extent, receive the same treatment, in the future
            return model_analysis
        else:
            print("Please, select an avaliable model, or use the standard, by giving no input")

    def testing(self,source):
        if self.model_type == "YoloID8n": ###Making sure the model does exist
            model_analysis = self.model_in_use.testing(source) ###The model was already created in post_init. New models should, by extent, receive the same treatment, in the future
            return model_analysis
        else:
            print("Please, select an avaliable model, or use the standard, by giving no input")




### YOLO ###
@dataclass
class YoloID8n():
    # YOLO('yolov8s.pt')            # Small
    # YOLO('yolov8m.pt')            # Medium  
    # YOLO('yolov8l.pt')            # Large
    # YOLO('yolov8x.pt')            # XLarge (mais preciso, mais lento)
    model = YOLO('yolov8n.pt') ### You might use any of this
    
    def __call__(self,source):
        
        tracker_result = self.model.track(
        source=source,
        tracker='botsort.yaml',  # or 'botsort.yaml'
        show=False,
        persist=True,
        save=False,
        classes=[0],       # Filter classes: 0=person
        conf=0.5,          # Confidence threshold
        iou=0.5,           # IOU threshold
        device='cpu',      # 'cpu' or 'cuda' (GPU)
        verbose=False      # Show logs
        )
        temporary_persons = []
        for frame in tracker_result: ### By definition, there should be ONLY one frame. However, in the possibility of having more than one, this implementation was done. It's important to note that it will leave ALL detections in the same list. That is, there will be no frame differentiation.
            element=frame.boxes
            try:
                for i in range(len(element.id)): ###Getting the number of detections
                    t_person = TempPerson()
                    t_person.id = element.id[i]
                    t_person.bb = element.xyxy[i]
                    t_person.confidence = element.conf[i]
                    temporary_persons.append(t_person)
            except:
                pass
        return {"result" : tracker_result, "temporary_persons" : temporary_persons}
    
    def testing(self, source):
                
        tracker_result = self.model.track(
        source=source,  #
        tracker='botsort.yaml',  # or 'botsort.yaml'
        show=False,
        save=False,
        classes=[0],  # Filter classes: 0=person
        conf=0.5,           # Confidence threshold
        iou=0.5,            # IOU threshold
        device='cpu',      # 'cpu' or 'cuda' (GPU)
        verbose=True        # Show logs
        )
        temporary_persons = ["testing"]
        return {"result" : tracker_result, "temporary_persons" : temporary_persons}



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

    id_system = IdSystem()
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
        else:
            saida = id_system(frame)
            tracker_result, temporary_persons = saida["result"], saida["temporary_persons"]
        
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
    
        