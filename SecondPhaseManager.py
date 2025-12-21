from dataclasses import dataclass, field
from typing import Any
from queue import Queue
import threading
import time

from SecondProcessManager import SecondProcessManager

@dataclass
class SecondPhaseManager(): ### Way more linear than phase 1
    SLEEP_TIME              : float = 0.000001
    SKIP_PERMANENCE         : int   = 0
    SKIP_MOVEMENT           : int   = 0 ### REMEMBER, THIS IS RUN ONLY WHEN PERMANENCE IS NOT SKIPPED. WHICH MENAS THAT IT WILL SKIP SKIP_PERMANENCE*SKIP_MOVEMENT
    SKIP_LINE               : int   = 0
    ### ideally, there's no reason to skip the skipper_buster 
    """
    element in input queue should have the following format {"frame" : frame, "model_analysis" : model_analysis, "reid_result" : list_of_temporary_person}
    
    element in output queue will have the format
    {"frame" : frame,
    "model_analysis" : model_analysis,
    "reid_result" : list_of_temporary_person,    
    "return_from_permanence_watcher" : return_from_permanence_watcher,
    "return_from_movement_watcher" : return_from_movement_watcher,
    "return_from_line_watcher" : return_from_line_watcher}
    """
    queues_from_first_phase : list = field(default_factory=list) ###Receives from first phase
    second_process_managers : list = field(default_factory=list) ###Internal
    output_queues           : list = field(default_factory=list) ###External

    def __post_init__(self):
        for i in range(len(self.queues_from_first_phase)):
            second_process_manager = SecondProcessManager(SKIP_PERMANENCE = self.SKIP_PERMANENCE, SKIP_MOVEMENT = self.SKIP_MOVEMENT, SKIP_LINE = self.SKIP_LINE) ###cria um second_process_manager para cada queue de entrada
            self.second_process_managers.append(second_process_manager)
            self.output_queues.append(Queue())
        
    def run_second_process(self,pos):
        ### IN - OUT ###
        local_queue = self.queues_from_first_phase[pos]
        local_out_queue = self.output_queues[pos]
        ### PROCESS ###
        local_second_process_manager = self.second_process_managers[pos]
        ### running process
        while True:
            time.sleep(self.SLEEP_TIME)
            if not local_queue.empty():
                element = local_queue.get_nowait()
                list_of_temporary_person = element["reid_result"]
                frame = element["frame"]
                frame_shape = frame.shape
                return_from_permanence_watcher, return_from_movement_watcher, return_from_line_watcher = local_second_process_manager(list_of_temporary_person,frame_shape)
                element["return_from_permanence_watcher"] = return_from_permanence_watcher
                element["return_from_movement_watcher"] = return_from_movement_watcher
                element["return_from_line_watcher"] = return_from_line_watcher
                if element: ###only if not empty
                    local_out_queue.put(element)
    
    
    
    def start(self):
    # Iniciar threads
        for i in range(len(self.second_process_managers)):
            thread = threading.Thread(
            target=self.run_second_process,
            args=(i,) ###args are the source, and the queue
            )
            thread.daemon = True ###Doesn't stop the program from ending
            thread.start() ###Create the thread
    
    def __call__(self):
        self.start()
        return self.output_queues
        

"""
Manager of Phase 2:
entry: output_from_phase_1 = number_output_queues, queues_from_sources, ID_processed_queues, REID_processed_queues, self.output_queues
element in output_queue = {"frame" : frame, "model_analysis" : model_analysis, "reid_result" : list_of_temporary_person}

PermanenceWatcher(element[list_of_temporary_person]) => list_of_temporary_person_permanence_watcher(that stayed in frame for the given amount : defined in PermanenceWatcher.py)
   ||||||
MovementWatcher(list_of_temporary_person_permanence_watcher) => list_of_temporary_person_movement_watcher(largest group moving in same direction)
   ||||||
LineFinder/Skipper_buster(list_of_temporary_person_movement_watcher) => list_of_people_in_line + list_of_skippers


"""