from dataclasses import dataclass, field
import numpy as np
import cv2
import time
import gc

from queue import Queue
from typing import Any, Tuple

from TempPerson import TempPerson

@dataclass
class DoomCounter():
   queues_to_check           : list[Queue] = field(default_factory=list)
   CYCLES_TO_ACTION          : int   = 100
   SLEEP_TIME                : float = 0.000_001
   waiting_multiplier_normal : int = 1_000 ###VERY ocasional
   print_queue_stats         : bool = False
   print_counting            : bool = False
   
   def __call__(self):
        doom_counter = 0
        doom_flag    = 0
        waiting_multiplier = 0 + self.waiting_multiplier_normal
        try:
            while True:
                time.sleep(self.SLEEP_TIME*waiting_multiplier)
                ### breaking mechanism - stops the program once all queues are empty. Keep in mind that second_phase barely uses queues ###
                if doom_counter % self.CYCLES_TO_ACTION == 0: ###printing takes a lot of time. Do it only for important values
                        ###ACTIONS TO COMPLETE###
                        if self.print_counting:
                            print ("doom counter :",doom_counter) ### see if the process is getting to the end
                        gc.collect()
                if doom_counter == 1000:
                    try: ###checking for empty queues
                        if all(q.empty() for q in self.queues_to_check):
                            doom_flag+=1
                            if doom_flag == 1:
                                print("\nDoom is aproaching\n")
                            elif doom_flag == 2:
                                print("\nDoom is INEVITABLE!\n")
                            elif doom_flag == 3:
                                print("\nDOOOOOOOOOOM!!!\n")
                            waiting_multiplier = 1
                        elif all(q.empty() for q in self.queues_to_check[:-1]): ###which menas that there's only output to process
                            waiting_multiplier = 1
                        else:
                            doom_flag=0
                            waiting_multiplier = 0 + self.waiting_multiplier_normal
                        if self.print_queue_stats:
                            print(
                                "\n\n============================================="
                                "\nobjects in queue_from_source", self.queues_to_check[0].qsize(),
                                "\nobjects in queue_from_id", self.queues_to_check[1].qsize(),
                                "\nobjects in reid_processed_queues", self.queues_to_check[2].qsize(),
                                "\nobjects in first_phase_output", self.queues_to_check[3].qsize(),
                                "\nobjects in output second phase", self.queues_to_check[4].qsize(),
                                "\n============================================="
                                "\n\n"
                            )
                    except Exception as e:
                        print("Erro: ",e)
                        
                    doom_counter = 0 ### reset doom counter
                doom_counter+=1
                
                if doom_flag>=3 :
                    break
                ### breaking mechanism - END ###            
        except Exception as e:
            print("ERRO IN DOOM:" , e)
        except KeyboardInterrupt:
            print("DOOM interrupted")


@dataclass
class Log():
    file : str = "log.txt"
    
    def __post_init__(self):
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
    fps: int
    width: int
    height: int
    output_file: str = 'output.mp4'
    
    def start(self):
        if self.fps is None or self.width is None or self.height is None:
            print("VideoWriter not properly initialized with video properties")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_file, 
                fourcc, 
                self.fps, 
                (self.width, self.height)
        )
    def release(self):
        self.writer.release()

    def update_file_name(self,new_name : str):
        self.output_file = new_name
    
    def __call__(self, frame):
        if self.writer:
            self.writer.write(frame)
        else:
            pass
    
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


@dataclass
class SleepTime():
    standard_value : float = 0.000001
    actual_value   : float = 0 + standard_value

    def __call__(self):
        return self.actual_value
    def increase(self):
        self.actual_value += self.standard_value
    
    def decrease(self):
        self.actual_value = max(self.actual_value-self.standard_value,self.standard_value)
    
    def reset(self):
        self.actual_value = 0 + self.standard_value

    def zero(self):
        self.actual_value = 0