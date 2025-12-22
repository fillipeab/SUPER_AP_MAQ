from dataclasses import dataclass, field
from typing import Any, Tuple
from queue import Queue
import threading
import time
import cv2
import numpy as np
from TempPerson import TempPerson
###WRITE IN VIDEO/OUTPUT###

@dataclass
class ThirdPhaseManager():
    ###Entry
    output_queues         : list[Queue]       = field(default_factory=list)

    ###Internal process
    output_file_names     : list[str]         = field(default_factory=list)
    list_of_video_writers : list[VideoWriter] = field(default_factory=list)
    list_of_logs          : list[Log]         = field(default_factory=list)


    ###Drawer
    bbdrawer_in_line_colour = (0,255,0) ###colour : GREEN
    bbdrawer_skipper_colour = (255,0,0) ###colour : RED
    bbdrawer_in_line : Any = None
    bbdrawer_skipper : Any = None

    ###Listed Counter
    Print_Listed_Counter          : bool = True
    Print_Listed_Counter_interval : int = 50

    ###Time Sleep
    SLEEP_TIME = 0.000001

    def __post_init__(self):
        if not self.output_file_names:
            self.output_file_names = [f"output_{i}.mp4" for i in range(len(self.output_queues))]
        if not self.list_of_video_writers:
            self.list_of_video_writers = [VideoWriter(self.output_file_names[i]) for i in range(len(self.output_queues))]
        if not self.list_of_logs:
            self.list_of_logs = [Log(f"log_{i}.txt") for i in range(len(self.output_queues))]
        
        ###DEFINING BBOX drawer
        self.bbdrawer_in_line = BBoxDrawer(2,0,self.bbdrawer_in_line_colour)
        self.bbdrawer_skipper = BBoxDrawer(2,0,self.bbdrawer_skipper_colour)


    def run_third_process(self, pos : int):
        local_output_queue = self.output_queues[pos]
        local_video_writer = self.list_of_video_writers[pos]
        local_output_file_name = self.output_file_names[pos]
        if local_output_file_name:
            local_video_writer.update_file_name(local_output_file_name)
        local_log = self.list_of_logs[pos]


        listed_counter = 0
        while True:
            try:
                time.sleep(self.SLEEP_TIME)
                """
                element in output queue will have the format - after 1 and second phase
                {"frame" : frame,
                "model_analysis" : model_analysis,                                 : analysis from model/varies
                "reid_result" : list_of_temporary_person,                          : list[TempPerson]   
                "return_from_permanence_watcher" : return_from_permanence_watcher, : list[TempPerson]
                "return_from_movement_watcher" : return_from_movement_watcher,     : list[TempPerson]
                "return_from_line_watcher" : return_from_line_watcher}             : dict[id,status]
                """
                if not local_output_queue.empty():

                    if self.Print_Listed_Counter:
                        listed_counter+=1
                        if listed_counter % self.Print_Listed_Counter_interval == 0:
                            print("Listed Counter: ",listed_counter)
                            listed_counter=0

                    ###gets the results that we want to see
                    element = local_output_queue.get_nowait()
                    ###gets the results that we want to see
                    model_analysis = element["model_analysis"]
                    return_from_permanence_watcher = element["return_from_permanence_watcher"]
                    return_from_movement_watcher = element["return_from_movement_watcher"]
                    result = model_analysis["result"] ### !!!!!!!!!!!!!!!!!!!!!!! This line is sensitive to the model type !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    
                    ### LOG so we can see who was classified as moving together
                    for temp_person in return_from_permanence_watcher:
                        local_log.write_in_log(("from permance watcher",str(temp_person.id)))
                    for temp_person in return_from_movement_watcher:
                        print("SPECIAL\n",temp_person)
                        local_log.write_in_log(("from movent watcher",str(temp_person.id)))
                    
                    
                    ### VIDEO WRITER ###

                    frame_to_write = result[0].plot() ###writes the frame, altered by YOLO, in a video
                    
                    ### COMPARISON - REID with dict coming from  return_from_line_watcher ###
                    list_of_in_line  = []
                    list_of_skippers = []                    ###filthy skippers!!!
                    temp_person_list = element["reid_result"] ### list of people to check
                    dict_from_line_watcher = element["return_from_line_watcher"]
                    if dict_from_line_watcher:
                        for temp_person in temp_person_list:
                                if temp_person.id in dict_from_line_watcher: ###line watcher findings
                                    status=dict_from_line_watcher[temp_person.id]
                                    if status == "in_line":
                                        list_of_in_line.append(temp_person)
                                    elif status == "skipper":
                                        list_of_skippers.append(temp_person)
                                    else:
                                        pass
                    ### COMPARISON - END ###

                    ### Writing in frame ###
                    if list_of_in_line:
                        frame_to_write = self.bbdrawer_in_line(frame_to_write,list_of_in_line) ###Marks in line
                    if list_of_skippers:
                        frame_to_write = self.bbdrawer_skipper(frame_to_write,list_of_skippers) ###Marks skippers
                    ### Writing in frame - END ###
                    local_video_writer(frame_to_write)
                    ### VIDEO WRITER - END ###
            except Exception as e:
                print("Exception in register :", e)
            finally:
                for video_writer in self.list_of_video_writers:
                    del video_writer
            break


    def __call__(self):
        for i in range(len(self.output_queues)):
            self.run_third_process(i)
            print("Processo de impressão iniciado")




@dataclass
class Log():
    file : str = "log.txt"
    
    def __post_in_init__(self):
        with open(self.file, 'w') as f:
            f.write("LOG OF OPERATION\n")

    def update_file_name(self,new_name : str):
        self.file = new_name
    
    def write_in_log(self, text):
        with open(self.file,'a') as f:
            f.write(f"{text}\n")

    def write_list_in_log(self,list):
        for e in list:
            self.write_in_log(str(e))
        self.write_in_log("\n")




@dataclass
class VideoWriter:
    output_file: str = 'output.mp4'
    fps: int = 30
    width: int = 1920
    height: int = 1080
    
    def __post_init__(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_file, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
    
    def update_file_name(self,new_name : str):
        self.output_file = new_name
    
    def __call__(self, frame):
        self.writer.write(frame)
    
    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.release()


@dataclass
class BBoxDrawer:
    thickness : int = 2
    padding   : int = 0  # Pixels para expandir
    color: Tuple[int, int, int] = (0, 255, 0)
    
    def __call__(self, frame: np.ndarray, list_of_temp_people: list[TempPerson]) -> np.ndarray:
        h, w = frame.shape[:2]
        
        for t_person in list_of_temp_people:
            bbox = t_person.bb
            x1, y1, x2, y2 = map(int, bbox)
            
            # Expande a bbox
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, self.thickness)
        
        return frame
