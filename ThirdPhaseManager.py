from dataclasses import dataclass, field
from typing import Any, Tuple
from queue import Queue
import threading
import time
import cv2
import numpy as np

from TempPerson import TempPerson
from DoomCounter_and_auxiliaries import BBoxDrawer, VideoWriter, Log, SleepTime

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
    bbdrawer_in_line : BBoxDrawer = field(default_factory=BBoxDrawer)
    bbdrawer_skipper : BBoxDrawer = field(default_factory=BBoxDrawer)

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

        sleep_time = SleepTime(self.SLEEP_TIME)
        listed_counter = 0
        local_video_writer.start()
        try:
            while True:
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

                    sleep_time.decrease() ###Reset it, slowly
                else:
                    sleep_time.increase()
                time.sleep(sleep_time())

        except Exception as e:
            print("Exception in register :", e)
        finally:
            local_video_writer.release()


    def start(self):
    # Iniciar threads
        for i in range(len(self.output_queues)):
            thread = threading.Thread(
            target=self.run_third_process,
            args=(i,) ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread

    def end(self):
        for videowriter in self.list_of_video_writers:
            videowriter.release()

    def __call__(self):
        self.start()